# MoE-1B: A 1B Dense-Equivalent Sparse Mixture-of-Experts Model

## Overview

`MoE-1B` is a Transformer decoder (causal LM) whose *active* compute per token matches a ~1B-parameter dense model, while its *total* parameter count is ~4.5B thanks to sparse routing. Following the GShard/Switch/Mixtral line of work, the standard MLP block of each layer is replaced with a Mixture-of-Experts (MoE) block: a set of 16 parallel expert FFNs, of which 2 are selected per token by a learned router.

| Config | Value |
|---|---|
| d_model | 2048 |
| num_layers | 24 |
| num_heads / head_dim | 16 / 128 |
| vocab_size (shared in/out embeddings) | 50,000 |
| num_experts | **16** |
| top_k (active experts) | **2** |
| expert FFN intermediate | 2560 |
| **Routing strategy** | **Learned softmax top-2** |
| **Capacity factor** | **1.25** |
| **Auxiliary loss** | **Load-balancing aux loss, α = 0.01** |
| Dense-equivalent (active) params | ≈ 1.01B |
| Total params | ≈ 4.53B |

---

## Parameters

Shared embedding (counted once, used for both input and output logits) plus per-layer attention and MoE blocks.

| Component | Params / layer | × Layers | Total |
|---|---|---|---|
| Embeddings (50,000 × 2048) | — | 1 | 102,400,000 |
| Attention (QKV 2048·6144 + O 2048·2048) | 16,777,216 | 24 | 402,653,184 |
| Expert FFNs (16 × [gate 2048·2560 + down 2560·2048]) | 16 × 10,485,760 = 167,772,160 | 24 | 4,026,531,840 |
| Router (2048 × 16) | 32,768 | 24 | 786,432 |
| LayerNorms (RMSNorm) | ≈ 0 | 24 | 0 |
| **Total params** | | | **4,532,371,456 ≈ 4.53B** |

**Active (dense-equivalent) params per token:**

| Component | Active params |
|---|---|
| Embeddings | 102,400,000 |
| Attention (always active) | 402,653,184 |
| Active experts (24 × 2 × 10,485,760) | 503,316,480 |
| Router (24 × 32,768) | 786,432 |
| **Active total** | **1,009,156,096 ≈ 1.01B** |

**Explicit math.** Per-expert FFN = `2 × 2048 × 2560 = 10,485,760`. Per layer, 16 experts = `16 × 10,485,760 = 167,772,160`; × 24 layers = `4,026,531,840`. Active expert cost = `2/16 × 4,026,531,840 = 503,316,480`. Total active = `102,400,000 + 402,653,184 + 503,316,480 + 786,432 = 1,009,156,096 ≈ 1.01B` → the **1B dense-equivalent** target. FLOPs/token ≈ `2 × active ≈ 2.02 GFLOP`, matching a 1B dense model. Sparse utilization = active/total ≈ 22%.

**Key digits:** total params **4.53B** · active params **1.01B** · num_experts **16** · top_k **2** · experts-per-layer **16** · router overhead **0.79M (0.02%)**.

---

## Routing Choice: Learned Softmax Top-2

Chosen: **learned top-2 with softmax-weighted combination**.

For each token, the router computes logits `hᵀ·W_router` (W_router: 2048×16), applies softmax over the 16 experts, selects the two highest-probability experts, and forms the output as the probability-weighted sum of their FFN outputs (weighted top-2, per GShard). If an expert exceeds its capacity slot, the token is dropped (computed as a residual skip connection); the aux loss pushes the router toward balance so drops stay rare.

Why top-2 over the alternatives:

- **vs. learned top-1:** Top-1 halves router state, but a single expert per token is strictly less stable — the router has far fewer gradient examples per expert per step, load balancing is harder, and top-1 models are measurably more prone to expert collapse and load imbalance (Switch Transformer reports top-2/top-3 beating top-1 at equal active cost in several settings). Top-2 costs only `+1 expert = +0.5× active MLP FLOPs` (still ≈ 1.01B active) for a meaningful quality/balance gain.
- **vs. learned soft (all experts, no top-k):** Full soft routing activates all 16 experts per token → 8× the MLP FLOPs, i.e. a dense model, defeating sparsity entirely. No.
- **vs. random/fixed routing:** Drops router capacity utilization and cannot specialize; it is only a baseline, not a real option for a quality-focused 1B-eq model.

**Capacity factor = 1.25.** Batch expert capacity = `(tokens_per_batch / 16) × 1.25`, giving 25% headroom over the ideal uniform split. With top-2, expected load is already even, so CF = 1.25 keeps token-dropping near zero while wasting only ~6% of expert compute — better than CF = 1.0 (dropping under mild imbalance) or CF ≥ 1.5 (squandered FLOPs).

**Auxiliary loss = 0.01.** A GShard/Switch-style load-balancing loss `L_aux = α · Σ N_experts · f_i · P_i` (f_i = fraction of tokens routed to expert i, P_i = mean router probability for i), with **α = 0.01**, added to the cross-entropy loss. This gradient-pushes the router toward uniform utilization; α = 0.01 is the standard magnitude that balances balance vs. harming token-level routing quality.

---

## Training Implications

- **Batch/tokens-per-batch must be a multiple of num_experts** (ideally 16), and batch size should be large enough that each expert sees ≥ 1 token per batch for stable router gradients; 1024+ tokens per batch is recommended.
- **All-to-all communication** (expert-parallel data shuffling) dominates communication; top-2 doubles the volume vs. top-1. On TPU/GPU pods, use expert-parallel with a token-dispatch pass; sequence packing plus capacity slots avoids padding waste.
- **Lower throughput per step than dense:** total activations across 16 experts raise memory/bandwidth despite sparse FLOPs; expect to tune microbatch and use activation checkpointing.
- **Aux loss must be tuned per scale** — α = 0.01 is a starting point, not a constant; monitor the `f_i` distribution early (step ≤ 500) to catch collapse before it entrenches.
- **Optimizer and data: same as a 4.5B dense model** (AdamW, bf16/mixed precision, gradient clipping) — optimizer state and checkpoint size track *total* params (≈ 4.5B), so memory is a 4.5B model's budget even though FLOPs are 1B's.
- **Mixture dropout and expert dropout** should be applied (and annealed) to avoid over-reliance on a few experts during pretraining.

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Router load imbalance / expert collapse | Some experts idle, others overflow → dropped tokens, wasted capacity | Load-balancing aux loss (α = 0.01), monitor `f_i`, CF = 1.25 headroom, expert dropout |
| Token dropping | Lossy when capacity exceeded → degraded quality | Capacity factor > 1 (1.25); treat drops as residual skip; track drop rate (< 1% target) |
| Communication overhead | All-to-all dispatch × top-2 can bottleneck at scale | Expert parallelism, padded dispatch buffers, overlap comm/compute |
| Aux loss hurts routing quality | Over-aggressive balancing degrades per-token routing accuracy | Keep α low; validate quality vs. balance trade-off ablations |
| Knowledge/quality gap vs. dense | Sparsity can lag dense quality at matched active params for small-scale runs | Top-2 (not top-1), large expert count for capacity; verify on small ablation before scale-out |
| Checkpoint/optimizer memory | Tracks 4.5B total params, not 1B | Budget accordingly; bf16, ZeRO/FSDP, and optional expert-parameter offload |
| Router logits temperature | Overconfident probabilities can cause token dropping | Optional router temperature (T ≈ 0.3) during training; noiseless top-k at inference |

---

**One-line summary:** 16 experts, top-2 learned softmax routing, CF 1.25, aux loss α 0.01 → **1.01B active / 4.53B total params**, delivering 1B-dense FLOPs with 4.5× the parameter capacity.