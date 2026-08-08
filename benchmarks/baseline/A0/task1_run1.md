The directory is empty except for `opencode.jsonc`, so I'll create the architecture document as a new markdown file here.
I don't have file-write tools available in this environment, so here's the architecture document directly:

---

# 1B Dense-Equivalent MoE — Architecture Document

## Overview

A decoder-only transformer in which every FFN block is replaced by a **Mixture-of-Experts (MoE) layer** with **8 experts and top-2 routing**. The model is sized so that **~1.0B parameters are active per token** (matching the per-token FLOPs of a 1B dense model) while holding **~2.9B parameters in total**.

- **Total params ≈ 2.86B** — all weights resident in memory.
- **Active params ≈ 1.01B/token** — the "dense-equivalent" figure (~35% of weights exercised per token).
- **FLOPs/token ≈ 2 × active ≈ 2.03 GFLOPS** (forward), i.e., compute parity with a 1B dense model.
- **Expert count:** 8 · **Active experts (top-k):** 2 · **Router:** learned linear gate + softmax, hard top-2 · **Capacity factor:** 1.25 · **Aux loss:** Switch-style load-balancing loss (α=0.01) + router z-loss (β=0.001).

| Config | Value | | Config | Value |
|---|---|---|---|---|
| Vocab `V` | 50,000 (tied embedding) | | Layers `L` | 24 (all MoE) |
| `d_model` | 1,792 | | Heads / head_dim | 28 / 64 |
| Experts `N` | 8 | | Top-k | 2 |
| Expert FFN width `d_e` | 3,584 (= 2×d) | | Activation | GELU |
| Capacity factor `c` | 1.25 | | Drop policy | token-drop (GShard) |

## Parameters

Per-layer math (`d=1792`, `d_e=3584`, `N=8`):

- **Attention (Q,K,V,O):** `4·d² = 4 × 1792² = 12,845,056`
- **Router:** `d·N = 1792 × 8 = 14,336`
- **Experts (8):** `N·(2·d·d_e) = 8 × (2 × 1792 × 3584) = 8 × 12,845,056 = 102,760,448`
- **LayerNorms (pre + post):** `2 × 2·d = 7,168`

| Component | Formula | Per layer | × Layers | Total |
|---|---|---|---|---|
| Tied embedding | `V·d` | — | — | 89,600,000 |
| Attention | `4·d²` | 12,845,056 | 24 | 308,281,344 |
| Router | `d·N` | 14,336 | 24 | 344,064 |
| MoE experts (8) | `8·(2·d·d_e)` | 102,760,448 | 24 | 2,466,250,752 |
| LayerNorms | `2·(2·d)` | 7,168 | 24 + final | 175,616 |
| **Total** | | | | **2,864,651,776 ≈ 2.86B** |

**num_experts = 8** · **top_k = 2**

**Active params per token (nominal, ideal top-2, k/N = ¼):**

```
embedding  = 89,600,000
attention  = 308,281,344
experts    = (2/8) × 2,466,250,752 = 616,562,688
Active     = 89,600,000 + 308,281,344 + 616,562,688
           = 1,014,444,032 ≈ 1.01B  ✔ (dense-equivalent)
```

**Worst case (capacity factor 1.25):** `89,600,000 + 308,281,344 + 1.25×616,562,688 ≈ 1.17B` → ≈ 2.34 GFLOPS/token. The design keeps the ≈1B claim with margin.

**Sparsity ratio:** `active/total ≈ 1.01/2.86 ≈ 35%`.

## Routing Choice

**Selected: learned top-2** (linear router: token hidden → 8 logits → softmax → hard top-2 selection; GShard/Mixtral-style). Rationale against the alternatives:

- **Learned vs. fixed/hash:** A hashing router (zero params, perfect balance) cannot learn semantic affinity between tokens and experts; it systematically underperforms a learned gate at equal compute. The 14K/router params are negligible (0.01% of total). → *learned.*
- **Top-2 vs. top-1:** Top-2 adds only one extra expert execution per token (expert FLOPs go from ⅛ → ¼ of the expert bank) but materially improves quality, gradient smoothness, and expert utilization; it resists the premature collapse that makes top-1 routers fragile, and is the standard choice (Mixtral, GShard, ST-MoE) at this scale. → *top-2.*
- **Top-2 vs. soft mixing:** A fully soft router (soft-MoE-style, weighted sum over all experts) eliminates sparsity, activating all 8 experts' compute per token — i.e., it becomes a *dense* model with 8× FFN params. That defeats the defining "dense-equivalent FLOPs" requirement. Hard top-2 is the deliberate trade: we accept a load-balancing problem (handled via aux loss + capacity factor) in exchange for the ~4× expert-compute savings. → *hard top-2.*

**Routing parameters:** learned linear gate · softmax over 8 · top-2 hard selection · capacity factor **1.25** · auxiliary load-balancing loss **α = 0.01** · router z-loss **β = 0.001**. No shared expert; all tokens route through exactly 2 of 8 experts every layer.

## Training Implications

- **Memory vs. FLOPs decoupling.** Optimizer state tracks the full 2.86B: BF16 weights 5.7 GB + grads 5.7 GB + AdamW fp32 (m,v) 22.9 GB ≈ **34 GB**. FLOPs, however, scale with the 1.01B active set. Use FSDP/ZeRO-3 + **expert parallelism (EP=8, one expert group per device)** to shard weights; 8×80GB A100s are ample.
- **Communication.** Each MoE layer does all-to-all: every token ships its 1792-dim hidden to ≤2 experts. Comm per token per layer is `2·d = 3,584` activations against `2·d·d_e ≈ 6.4M` expert MACs → **communication-to-compute ratio is low (~1/d_e)**; overlap the all-to-all with attention compute.
- **Load balancing is an explicit training signal.** Train with the Switch aux loss (α=0.01) on `fraction-routed × mean-router-prob` and the z-loss (β=0.001) to bound router-logit growth (Mixtral-style stability). Monitor routing entropy and per-expert utilization; anneal α only after loss converges.
- **Capacity & dropping.** With CF=1.25, if a batch overflows an expert buffer, excess tokens are dropped (GShard policy). Track drop rate as a first-class metric (target < 0.5%); if it rises, raise CF or switch to ST-MoE "no-drop + P-drop".
- **Training dynamics.** Same LR schedule/warmup as a dense 1B model. Expect the usual MoE Pareto: lower loss per FLOP than a 1B dense model, but wall-clock can lag at this scale because all-to-all overhead is not amortized until batch sizes grow. Prefer larger micro-batches over many small ones to keep all 8 expert groups busy.
- **Inference.** KV-cache is attention-only (308M) → small cache. Decode FLOPs ≈ a 1B dense model, but weights are 2.86B, so memory-bandwidth-bound decoding needs a reasonable batch to keep all expert groups utilized.

## Risks

1. **Router collapse / dead experts** — one or two experts hog all tokens. Mitigated by aux loss + z-loss + CF; watch "dead-expert fraction" and routing entropy each step.
2. **Silent quality loss via token dropping** — overflow drops are invisible to loss unless measured; enforce a drop-rate budget and validate on a routing-stress eval set.
3. **Training instability** — growing router logits (rank collapse) destabilizes loss; z-loss, gradient clipping, and lower peak LR are the primary levers.
4. **Communication erodes the FLOPs advantage** — at 1B-active scale with 8 experts, all-to-all per layer can make wall-clock throughput *worse* than a dense 1B model; measure tokens/s, not just FLOPs.
5. **Weak expert specialization at this scale** — ST-MoE-style evidence shows specialization is limited when experts are small; if routing never diversifies, the aux-loss/CF machinery adds overhead without payoff. Contingency: fewer experts (e.g., 4) or a soft-router variant.
6. **CF × aux-loss interaction** — an aggressive aux loss "solves" balance by smearing tokens uniformly, which can hurt capacity utilization; tune jointly with CF and drop rate.
7. **Serving footprint** — 2.86B weights must be resident (or expert-offloaded); quantization of experts and batch scheduling are deployment concerns that a 1B dense model avoids.
8. **Batch/EP mismatch** — if the batch is too small for 8-way EP, expert groups idle: memory is wasted and the sparse FLOPs never materialize as throughput.

---

Numbers double-checked: `4·1792² = 12,845,056`; `8 × 2 × 1792 × 3584 = 102,760,448`; total `≈2.864B`; active `≈1.014B` (top-2, ¼ of experts), `≈1.17B` worst-case at CF 1.25.