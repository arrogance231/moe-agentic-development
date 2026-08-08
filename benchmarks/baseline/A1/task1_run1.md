# MoE-1B-Dense-Equivalent — Architecture Document

**Design goal:** match the per-token FLOPs / inference compute of a ~1B-parameter dense transformer, while scaling total capacity through sparse expert routing.

---

## 1. Overview

A 24-layer decoder-only transformer (`d_model=2048`, 32 heads, RoPE), where every FFN block is replaced by a **top-2 MoE FFN with 64 experts**. Total parameters ≈ **10.33B**; parameters active per token ≈ **0.97B** ≈ 1B dense-equivalent.

| Spec | Value |
|---|---|
| num_experts (E) | 64 |
| top_k (K) | 2 |
| Routing | Learned softmax top-2 (renormalized weighted average) |
| Expert FFN dim | 1024 (up / gate / down) |
| Capacity factor (C) | 1.25 |
| Aux loss | Load-balancing α=0.01 + router z-loss α_z=0.001 |
| Vocab / d_model / heads | 128k / 2048 / 32 |
| Total parameters | **10,331,717,632 (≈10.33B)** |
| Active parameters/token | **≈969,940,992 (≈0.97B)** |

---

## 2. Parameters

**Per-expert FFN:** `3 × 2048 × 1024 = 6,291,456` (up + gate + down)
**All experts / layer:** `64 × 6,291,456 = 402,653,184`
**Experts total:** `24 × 402,653,184 = 9,663,676,416`

| Component | Formula | Parameters |
|---|---|---|
| Embeddings (tied in/out) | 128,000 × 2,048 | 262,144,000 |
| Attention (24 layers) | 24 × 4 × 2,048² | 402,653,184 |
| Router gates (24 layers) | 24 × 2,048 × 64 | 3,145,728 |
| Expert FFNs (24 × 64) | 24 × 64 × 3 × 2,048 × 1,024 | 9,663,676,416 |
| RMSNorm weights | ≈24 × 3 × 2,048 | 147,456 |
| **Total** | | **10,331,717,632 ≈ 10.33B** |

**Active per token** (dense-equivalent): `262,144,000 + 402,653,184 + (24 × 2 × 6,291,456) + 3,145,728 + 147,456`
`= 969,940,992 ≈ 0.97B ≈ 1B` ✓

Total/active ratio ≈ 10.6×; with capacity factor 1.25 the realized active compute ≈ 1.2B-equivalent, still ~1B-class.

---

## 3. Routing choice

**Chosen: learned top-2 softmax router** — token-softmax over all 64 logits, take top-2, renormalize over the two selected, weighted sum of expert outputs.

| Option | Pros | Cons |
|---|---|---|
| top-1 (Switch/GShard) | Minimal expert compute | Coarse specialization; needs aggressive aux loss; load-imbalance prone |
| **top-2 (chosen)** | 2× gradient signal, smoother specialization, better quality/FLOP, tolerable imbalance | ~2× expert compute vs top-1 (still ~10% of dense) |
| Learned expert-choice (EC) | No token dropping; no aux loss needed | Requires token-level masking; token loss becomes "unrouted"; different gradient semantics |
| Soft (full average) | Max smoothness | Activates all experts → no sparse savings, compute = dense (defeats purpose) |

Justification: top-2 is the sweet spot — nearly dense quality at sparse cost, naturally balances load better than top-1, and gives each token two experts' features to co-adapt. EC's routing asymmetry is unnecessary at this scale where a small aux loss (α=0.01) plus capacity buffer (C=1.25) is sufficient.

**Capacity factor:** expected tokens/expert/layer with batch of 2,097,152 tokens = `(2,097,152 × 2) / 64 = 65,536`; buffer → capacity = `65,536 × 1.25 = 81,920`. Tokens beyond capacity are dropped (tracked; target <0.1%).

**Aux loss:** `L_aux = α · E · Σ_e f_e · P_e` (f_e = routed fraction, P_e = prob mass) with α=0.01; z-loss `α_z·Σ_t (log Σ_e e^{g_e})²` with α_z=0.001 for router entropy stability.

---

## 4. Training implications

- **Compute:** ~1B-equivalent FLOPs/token → can train ~1T tokens on the budget of a dense 1B model, but 10× parameter memory.
- **Optimizer/memory:** Adam state for 10.33B params (≈165 GB fp32) → requires FSDP/expert parallelism; experts sharded across GPUs (EP ≥ 8), TP+EP hybrid.
- **Communication:** All-to-all dispatch/combine per MoE layer; must overlap with compute and use grouped GEMMs; comms dominate at small batch sizes.
- **Batch size:** load imbalance scales as `1/√(B·S)`; need large tokens/batch (≥2M) or capacity violations rise.
- **Stability:** lower LR for router params, longer warmup, z-loss; monitor router entropy and dropped-token ratio.
- **Regularization:** expert dropout (e.g., 0.1); optionally start with a few layers dense and progressively activate MoE (warm-starting).
- **Tuning knobs:** (C, α, α_z, K, E) interact; log per-expert utilization histograms to a tracker.

---

## 5. Risks

- **Load imbalance / capacity overflow** → token dropping degrades quality on heterogeneous data (mitigate: raise C, strengthen aux loss).
- **Routing collapse** — few experts dominate; entropy decays (mitigate: z-loss, expert dropout, lower router LR).
- **Aux loss interference** — balancing penalty competes with task loss; over-weighted α regresses quality.
- **Underfilled experts** — 64×6.3M experts need enough data/pretrain tokens to specialize; risk of dead experts early on.
- **Communication overhead** — all-to-all can negate compute savings on small batches / few GPUs; single-GPU serving infeasible without expert offloading.
- **Representation fragmentation** — token representations split across expert subspaces reduce cross-expert co-adaptation (partially offset by K=2).
- **Serving complexity** — 10.3B checkpoint, expert CPU offload, dynamic routing hurt first-token latency vs dense 1B.
- **Hyperparameter sensitivity** — C/α/K/E tradeoffs are model-size-dependent; expect a re-tuning pass after scale-up.