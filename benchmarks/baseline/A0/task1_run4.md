# 1B Dense-Equivalent MoE Architecture

## Overview

A sparse Mixture-of-Experts (MoE) decoder-only transformer whose **per-token active parameter count (~1.17B)** and FLOP profile match a ~1B-parameter dense baseline, while storing ~6.35B total parameters. The dense MLP block of each layer is replaced by a set of 16 small expert FFNs; each token is routed to the top-2 experts by a learned router (token-choice, softmax). Model is non-causal here assumed — apply causal mask; layout follows standard LLaMA-style blocks (pre-RMSNorm, RoPE, GQA optional).

## Parameters

| Component | Formula | Per layer | Total (22 layers) |
|---|---|---|---|
| Embeddings (tied in/out) | `V × d = 32000 × 2048` | — | 65,536,000 |
| Attention (Q,K,V,O) | `4 × d² = 4 × 2048²` | 16,777,216 | 369,098,752 |
| Experts (16 × 3-proj FFN) | `E × 3 × d × d_exp = 16 × 3 × 2048 × 2736` | 268,959,744 | 5,917,114,368 |
| Router | `d × E = 2048 × 16` | 32,768 | 720,896 |
| **Total** | | | **6,352,470,016 ≈ 6.35B** |

**Active params per token** (top-2 routing):

```
Active = embeddings + attention + active experts + router
       = 65,536,000 + 369,098,752 + (22 × 2 × 16,809,984) + 720,896
       = 65,536,000 + 369,098,752 + 739,639,296 + 720,896
       = 1,174,994,944  ≈  1.17B
```

Per-expert FFN: `3 × d × d_exp = 3 × 2048 × 2736 = 16,809,984` (gate/up `d→d_exp`, down `d_exp→d`).

| Key numbers | Value |
|---|---|
| Dense-equivalent | ~1.0–1.2B (active) |
| Total parameters | **6.35B** (sparsity factor ≈ 5.4×) |
| `num_experts` | **16** (per layer) |
| `top_k` (active experts) | **2** |
| `expert_intermediate_dim` | 2,736 |
| `d_model` / layers / heads | 2,048 / 22 / 16 |
| FLOPs per token | ≈ 2 × active ≈ **2.35 GFLOPs** (≈ dense 1B) |

## Routing choice

**Learned top-2 (token-choice, softmax) router.**

- A linear layer `d → E` per layer produces logits; a softmax gives expert probabilities; the **top-2** experts are selected and the token's hidden state is split/reweighted by the normalized routing probabilities (weighted sum).
- **Why top-2 over top-1:** top-1 concentrates all gradient through a single expert, worsening load imbalance and hurting the parallel dispatch (no gradient through the second expert's load), while top-2 (Mixtral-style) gives the router a secondary gradient path, smoother training, and better effective capacity at the same active-FLOP budget.
- **Why learned over static:** hand/random assignment (`seq % E`) ignores token content and wastes expert specialization; a learned router lets experts self-organize by topic/syntax, and lets me add a load-balancing auxiliary loss.
- **Why soft (weighted) over hard (one-hot):** the top-2 weighted combination is differentiable and reduces variance of expert outputs; pure hard top-1 has dead-expert risk and no inter-expert blending.

**Routing configuration:**

| Setting | Value |
|---|---|
| Routing strategy | learned softmax, token-choice, weighted top-2 |
| `capacity_factor` | **1.25** (training), 1.0 (inference) |
| Auxiliary loss | load-balancing (Switch-style), coefficient `α = 0.01` |
| Auxiliary loss | `α × E × Σₑ fₑ·Pₑ` (per-layer, added to total loss) |

## Training implications

- **Compute-efficient:** ~2.35 GFLOPs/token vs ~6.35B total params → ~5.4× FLOP savings vs a dense 6.35B; data throughput ≈ dense 1.2B model.
- **Capacity factor 1.25** gives 25% slack so expert buffers don't drop tokens during training; this keeps the top-2 promise but inflates per-batch compute slightly; at inference drop it to 1.0 (token-choice still exact).
- **Load-balancing aux loss (α=0.01)** steers the router toward uniform expert utilization, preventing expert collapse/dead experts; keep α small so it doesn't override task loss.
- **Communication:** MoE layers need an all-to-all (token dispatch/gather) per layer — on TPU/GPU clusters this is the dominant scaling bottleneck; requires efficient group-geMM (e.g., GShard-style grouping) to realize the FLOP savings.
- **Optimization:** batch size must stay large (many tokens/expert) for expert gradients to be stable; use bf16 with standard LLM training (AdamW, warmup + cosine, weight decay on non-bias/norm params); router logits can be scale-stabilized to avoid "router collapse" at init.
- **Memory:** optimizer state scales with **total** params (6.35B), so VRAM cost is that of a 6.3B dense model even though compute is 1B — a key trade-off to plan for (ZeRO/FSDP sharding recommended).

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Router collapse / load imbalance | Experts die, capacity wasted, aux loss fights task loss | Load-balancing aux loss (α=0.01) + expert dropout + monitor histogram of routing probabilities |
| All-to-all comm bottleneck | Offsets FLOP gains at scale | Group-wise expert parallel, token grouping, overlap comm/compute |
| Memory/optimizer state for 6.35B params | VRAM > dense 1B despite compute parity | FSDP/ZeRO-3, possibly share expert weights across layers (extreme), or reduce E |
| `capacity_factor` 1.25 → some tokens always dropped under load spikes | Information loss during training | Use token-choice with padding (no drop) or raise capacity at cost of compute; validate top-2 actually dispatched |
| Small experts underfit complex tokens | Quality regression vs dense baseline | Tune d_exp / top_k; consider top-2→3 for hard tokens (extra compute) |
| Inference fragmentation / low batch util | Poor MFU for small batch sizes | Serve with expert-parallel batching, batch tokens across requests before dispatch |
| Aux loss too weak/strong | Dead experts or over-regularized routing | Grid-search α ∈ {0.001, 0.01, 0.1}; early-stopping router collapse metrics |

**Verified parameter math** (sum of rows): `65,536,000 + 369,098,752 + 5,917,114,368 + 720,896 = 6,352,470,016` ✓; active total `1,174,994,944` ≈ 1.17B ✓ — this is the 1B dense-equivalent budget.