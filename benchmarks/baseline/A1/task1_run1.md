# 1B Dense-Equivalent MoE — Architecture Document

## 1. Overview

A **Mixtral-style token-choice MoE** decoder-only transformer sized so that **active (dense-equivalent) parameters per token ≈ 1.03B**. All shared layers (embeddings, attention, norms, routers) are dense; only the FFN block is replaced by a sparse set of experts. Each token is routed to its top-2 experts; the rest of the expert parameters stay idle for that token.

| Design constant | Value |
|---|---|
| Hidden size `H` | 2,048 |
| Layers `L` | 22 |
| Vocabulary `V` (tied embeddings) | 50,000 |
| Num experts `E` | 16 |
| Active experts `top_k` | 2 |
| Expert FFN intermediate `I` (SwiGLU) | 2,048 |
| Capacity factor `CF` | 1.25 (train) / 1.0 (inference) |
| Aux loss | load-balancing (Switch-style), `α = 0.01` (+ z-loss `1e-4`) |

**Key figures:** `num_experts = 16`, `top_k = 2`, total parameters ≈ **4.90B**, active per token ≈ **1.03B** (dense-equivalent ≈ **1B**).

## 2. Parameters

Per-expert FFN is SwiGLU (`gate + up + down`, 3 matrices of `H×I`). Parameter math:

- Embeddings (tied): `50,000 × 2,048` = **102,400,000**
- Attention per layer: `QKV 3×2048² + O 2048²` = `12,582,912 + 4,194,304` = `16,777,216`; ×22 = **369,098,752**
- Router per layer: `2,048 × 16` = `32,768`; ×22 = **720,896**
- LayerNorm (2/layer): `8,192 × 22 + 2,048` = **182,272**
- Expert FFN per expert: `3 × 2,048 × 2,048` = `12,582,912`; ×16 experts ×22 layers = **4,429,185,024**
- **Total (all weights):** `102,400,000 + 369,098,752 + 720,896 + 182,272 + 4,429,185,024` = **4,901,586,944 ≈ 4.90B**

**Active per token:** shared (`472,401,920`) + `top_k × expert` = `2 × 12,582,912 × 22` = `553,648,128` → **≈ 1.03B** (~1B dense-equivalent).

| Component | Params (digits) | Active/token |
|---|---|---|
| Embeddings (tied) | 102,400,000 | 102,400,000 |
| Attention (22 layers) | 369,098,752 | 369,098,752 |
| Router (22 layers) | 720,896 | 720,896 |
| LayerNorm | 182,272 | 182,272 |
| Expert FFNs (16 × 22) | 4,429,185,024 | 553,648,128 (2 experts) |
| **Total** | **4,901,586,944 (≈4.90B)** | **1,026,050,048 (≈1.03B)** |
| **num_experts / top_k** | **16 / 2** | — |

Only **12.5%** of expert capacity is used per token (2/16), roughly halving FFN compute vs. a dense 1B (dense FFN would be `3·2048·8192` per layer).

## 3. Routing choice

**Top-2, learned, soft (token-choice), gated via softmax** — the Mixtral design.

- **top-2 over top-1:** two experts give a richer mixture, better gradient flow into multiple experts, and more robust performance. Top-1 (Switch) halves expert compute and simplifies load balance but relies on a single expert per token and trains slower per step at equal width.
- **Learned over static:** router is a trained `Linear(H, E)` — no hand-crafted assignment rules; adapts to data.
- **Soft over hard:** expert outputs are weighted by softmax gate probabilities (`softmax(top-2 logits)`), making routing differentiable and gradients flow into both experts. No hard (0/1) selection, no straight-through estimator needed.
- **Token-choice over expert-choice:** each token independently picks its top-2 experts. Simpler, matches Switch/Mixtral, and pairs cleanly with a capacity factor that bounds worst-case work per expert.

**Load balance:** capacity factor `CF = 1.25` during training (`⌈top_k · N_tokens / E⌉ · CF` tokens per expert; dropped tokens masked) and `CF = 1.0` at inference. Plus a **Switch-style load-balancing aux loss** on router logits (`α = 0.01`) that minimizes the KL between empirical expert load and uniform, encouraging balanced utilization without hard top-1 constraints. A small **z-loss** (`1e-4`, ST-MoE) keeps router logits from growing unbounded and avoids instability.

## 4. Training implications

- **Batch / sequence:** with 50k vocab and H=2048, use ~15B–50B tokens at bf16; route at the *token* granularity with a minimum per-expert batch to avoid noisy routing updates.
- **Parallelism:** expert parallelism (each of 16 experts on one device or sharded) + all-to-all token dispatch per layer; at this scale the compute/communication ratio is tight — overlapping all-to-all with attention/FFN is mandatory.
- **Router dynamics:** expect load drift early; warm-up the router LR and rely on the aux loss to prevent collapse. Monitor per-expert token counts and top-k selection frequency each step.
- **Regularization:** dropout on expert FFNs is **not** recommended (hurts expert specialization); instead rely on capacity-factor token dropping + aux loss.
- **Checkpointing/memory:** expert params (≈4.4B) are distributed; the dense-equivalent optimizer state is that of a ~1B model per rank, so memory is manageable relative to a true 4.9B dense model.
- **Layers:** every layer has its own router + 16 experts; no interleaving of dense/sparse layers needed at 22 layers.

## 5. Risks

- **Load imbalance / dead experts:** if the router collapses onto few experts, effective capacity drops below dense-equivalent and aux loss fights the objective. Mitigated by α tuning and `CF`.
- **Token dropping:** at `CF=1.25` some tokens get dropped during training (loss spike if too aggressive); raising `CF` reduces drops but increases wasted compute.
- **Communication overhead:** all-to-all at only 1B active compute means communication can dominate; small MoEs often show *worse* tokens/sec than dense 1B unless dispatch is heavily optimized. This is the #1 hardware risk at this size.
- **Router instability:** logits can grow large → softmax saturation; z-loss + logit clipping mitigates.
- **Evaluation sensitivity:** inference must use the same routing (and `CF=1.0`) as training; switchy behavior can cause eval variance.
- **Aux-loss interference:** the balance term can override expert specialization if α is too large; tune against downstream quality, not just balance.
- **Data-hungriness:** 16 experts need enough tokens per expert to specialize; on small corpora the extra parameters give diminishing returns.