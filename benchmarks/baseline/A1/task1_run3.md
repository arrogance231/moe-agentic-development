# 1B Dense-Equivalent MoE — Architecture Document

## 1. Overview

A sparse Mixture-of-Experts transformer designed to match the **per-token compute (FLOPs) of a ~1.07B-parameter dense model** while holding ~5.8B total parameters. Each dense feed-forward block is replaced with 16 experts; every token activates 2 experts (top-2), so the FFN FLOPs per token are unchanged versus the dense baseline.

| Config | Value |
|---|---|
| d_model | 2048 |
| num_layers | 20 |
| num_heads / head_dim | 32 / 64 |
| vocab_size | 32000 (RoPE, no learned position params) |
| Dense FFN intermediate (baseline) | 8192 |
| Expert intermediate | 4096 |
| num_experts | 16 (per layer) |
| top_k | 2 |
| Capacity factor | 1.25 |
| Auxiliary loss | Switch-style load-balancing, α = 1e-2 |

## 2. Parameters

Notation: `d = d_model`, `E = num_experts`, `k = top_k`, `I_e = expert intermediate`, `L = layers`.

| Component | Formula | Value | Params |
|---|---|---|---|
| Token embedding (tied w/ LM head) | `V × d` | 32000 × 2048 | 65,536,000 |
| Attention / layer | `4d²` | 4 × 2048² | 16,777,216 |
| Router / layer | `d × E` | 2048 × 16 | 32,768 |
| MoE FFN / layer | `E × 2d·I_e` | 16 × (2·2048·4096) | 268,435,456 |
| **Total per layer** | — | — | 285,245,440 |
| **Layers × 20** | — | — | 5,704,908,800 |
| **Total parameters** | + embedding | — | **≈ 5.77B** |
| **Active params / token** | emb + L·(attn + k·expert) | 65.5M + 20·(16.8M + 2·16.8M) | **≈ 1.07B** |

**FLOP-equivalence check (dense vs MoE):**
- Dense FFN per token: `2d · I_dense = 2·2048·8192 = 33.55M` MACs.
- MoE FFN per token: `k · 2d·I_e = 2 · 2·2048·4096 = 33.55M` MACs. ✅
- Full model: embedding + 20·(attention + FFN) ⇒ **1.07B active params ≈ 1B dense-equivalent**. ✅

## 3. Routing choice

**Learned top-2 softmax gating (token-level, per layer), with a load-balancing auxiliary loss.**

- **Top-1 rejected**: single-expert routing gives weak gradient signal to only one expert per token, slows expert specialization, and is more prone to routing collapse. Under the FLOP-equivalence constraint it also halves expert capacity utilization.
- **Top-3+ rejected**: increases compute per token (breaks the 1B dense-equivalent FLOP budget) with marginal quality gains at this scale.
- **Soft/all-expert routing rejected**: computing all 16 experts and weighting their outputs is not FLOP-equivalent to the dense baseline — it is effectively a 16× denser FFN, violating the design constraint and raising training cost ~8×.
- **Top-2 chosen**: doubles the number of experts receiving gradients per token vs top-1, is the de-facto standard (GShard, Mixtral), and with `I_e = I_dense / k` keeps FLOPs identical to dense. Router is a single linear layer `d → E` with per-token softmax over expert logits.

## 4. Training implications

- **Expert parallelism + all-to-all dispatch**: 16 experts/layer across ≥16 GPUs; token dispatch uses all-to-all (GShard pattern). Communication cost grows with batch size and `k`.
- **Same FLOPs, more params**: wall-clock per step ≈ dense (compute-bound), but memory footprint ≈ 5.8B params — requires sharding; activations are only for active experts.
- **Capacity factor 1.25**: each expert buffers `CF · (B·S·k/E)` tokens. Excess tokens are dropped and sent through a residual (untouched) path. Budget: ~20% headroom absorbs imbalance before aux-loss kicks in.
- **Auxiliary loss**: `α · N · Σ f_i · P_i` (Switch formulation, f_i = routed fraction, P_i = router prob, N = experts), α = 1e-2. This is the primary lever for load balance; tune α first.
- **Per-expert batch**: each expert sees ~B·S·2/16 = B·S/8 tokens — small experts need large global batches (≥ 512 sequences) or long runs to get adequate gradient signal.
- **Sequence packing + loss masking** to keep expert workloads even across the batch.
- **Stable init / fp8 training**: experts are independent copies of the dense FFN — a warm start from the pretrained dense checkpoint is cheap and accelerates convergence.

## 5. Risks

- **Load imbalance / dropped tokens**: if routing collapses onto few experts, dropped tokens degrade quality and wasted capacity grows. Mitigated by aux loss + capacity factor, but both add tunable hyperparameters.
- **Routing collapse**: the router can degenerate to uniform or single-expert assignments. Watch for `f_i` variance and the aux loss value during warmup.
- **Training instability**: sparse gating + small expert batches can cause high gradient variance. Use higher LR warmup, gradient clipping, and possibly z-loss to prevent logit drift.
- **All-to-all bottleneck**: at high batch sizes, communication can dominate and negate the FLOP advantage; overlapping dispatch with compute and using top-2 grouping helps.
- **Expert overfitting / under-specialization**: at 1B scale with limited training tokens, 320 expert FFNs risk memorization; if evaluation shows per-expert entropy dropping too fast, increase aux-loss α or reduce `E`.
- **No free lunch vs dense**: at small compute budgets, sparse models can underperform dense peers; a 1B-equivalent MoE only pays off with enough training tokens — otherwise a plain dense 1B model may win.

**Parameter math (explicit):** embedding `32000·2048 = 65.5M`; attention/layer `4·2048² = 16.8M`; MoE FFN/layer `16·2·2048·4096 = 268.4M`; per-layer `16.8M + 0.03M + 268.4M ≈ 285.2M`; `×20 = 5.70B`; total `+65.5M = 5.77B`. Active/token: `65.5M + 20·(16.8M + 2·16.8M) ≈ 1.07B ≈ 1B dense-equivalent`.