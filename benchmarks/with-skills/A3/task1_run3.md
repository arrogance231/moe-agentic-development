I'll load the architecture skill workflow and look for the referenced calculator tool.
The skills are inline reference material. Let me locate the calculator tool referenced by the skill.
No calculator file is present in this session, so I'll work the parameter math by hand from the skill's formulas. Here is the architecture document.

---

# MoE Architecture: 1B Dense-Equivalent MoE

## Overview

This design replaces a dense baseline of **0.96B parameters** (24 layers, `d_model` 1536, FFN multiplier 4) with a Mixture-of-Experts model of the same layer geometry using **64 experts, top-2**, one all-to-all router per layer. The MoE keeps attention, embeddings, and layernorms shared and dense, while each of the 64 experts carries its own full GLU FFN. Total parameters grow to **43.76B** (45.8× the dense baseline) but only **1.63B are activated per token** (1.71× the dense baseline), so compute stays near a 1B-class model. This matches the skill's reference profile (~1B dense-equivalent, 64 experts, top-2, 8 GPUs). Routing is **Top-2** with a **capacity factor of 1.25** and an **auxiliary load-balancing loss of 0.01**.

## Parameters

Fixed geometry: `num_layers = 24`, `d_model = 1536`, `ffn_mult = 4` → `ffn_dim = 6144`, `vocab = 32000`, `seq_len = 2048`, `num_experts = 64`, `top_k = 2`.

Per-layer terms:
```
attention  = 4 * d_model²                 = 4 * 2,359,296              = 9,437,184
ffn/experts = 3 * d_model * ffn_dim       = 3 * 1536 * 6144            = 28,311,552
layernorm  = 2 * d_model                  = 3,072
embedding  = vocab * d_model              = 32,000 * 1536              = 49,152,000
```

Dense baseline:
```
dense = 24 * (4*d_model² + 3*d_model*ffn_dim + 2*d_model) + vocab*d_model
      = 24 * (9,437,184 + 28,311,552 + 3,072) + 49,152,000
      = 24 * 37,751,808 + 49,152,000 = 906,043,392 + 49,152,000
      = 955,195,392  (~0.96B)
```

MoE totals:
```
moe      = 24 * (4*d_model² + 2*d_model + 64 * 3*d_model*ffn_dim) + vocab*d_model
         = 24 * (9,437,184 + 3,072 + 1,811,939,328) + 49,152,000
         = 24 * 1,821,379,584 + 49,152,000 = 43,713,110,016 + 49,152,000
         = 43,762,262,016  (~43.76B)

activated = 24 * (4*d_model² + 2*d_model + 2 * 3*d_model*ffn_dim) + vocab*d_model
          = 24 * (9,437,184 + 3,072 + 56,623,104) + 49,152,000
          = 24 * 66,063,360 + 49,152,000 = 1,585,520,640 + 49,152,000
          = 1,634,672,640  (~1.63B)

param_ratio      = 43,762,262,016 / 955,195,392 = 45.8×
activated_ratio  = 1,634,672,640  / 955,195,392 = 1.71×
```

| Component | Dense | MoE | Activated |
| --- | --- | --- | --- |
| Attention (all layers) | 226,492,416 | 226,492,416 | 226,492,416 |
| Expert FFNs (all layers) | 679,477,248 | 43,486,543,872 | 1,358,954,496 |
| Layernorms (all layers) | 73,728 | 73,728 | 73,728 |
| Embedding | 49,152,000 | 49,152,000 | 49,152,000 |
| **Total** | **955,195,392** | **43,762,262,016** | **1,634,672,640** |

**(num_experts = 64, top_k = 2)**

## Routing choice

- **Strategy: Top-2** — the standard training default. It lets each token blend two expert representations (a large quality gain over Top-1 at only ~2× the dispatch/all-to-all cost of Top-1) and is the best fit for a training-focused design.
- **top_k = 2** — keeps activated params at 1.63B (compute near the 1B dense baseline); top-1 would halve activated expert capacity and drop to 1.16B activated but sacrifice the knowledge-blending quality gain.
- **Capacity factor = 1.25** — absorbs token-routing imbalance without dropping tokens (1.0 would waste compute under skew); can be reduced to 1.0 once load is balanced.
- **Auxiliary loss = 0.01** — upper end of the 0.001–0.01 band; with 64 experts the router has a large entropy surface and needs a strong pull against collapse. Anneal toward 0.001 after warmup if routing stays balanced.

## Training implications

- **Compute:** `FLOPs/token = 6 * activated + 4 * num_layers * d_model * seq_len`. Dense = `6*955,195,392 + 301,989,888 = 6.03B`; MoE = `6*1,634,672,640 + 301,989,888 = 10.11B`. The MoE costs **1.68× the dense baseline FLOPs/token** — 43.76B total params but only 1.63B active, so compute tracks activated, not total, params.
- **Memory:** weights alone are 87.5 GB in BF16; AdamW optimizer states add ~525 GB. Shard with EP (experts) + ZeRO (attention/embedding/layernorm, gradients, optimizer). **EP must divide 64**; on 8 GPUs use e.g. EP=8/DP=1 (8 experts per rank) or EP=4/DP=2 (16 per rank) so that `DP*TP*PP*EP = 8`.
- **Parallelization:** one all-to-all per MoE layer → **24 all-to-alls per step**. Dispatch volume is `top_k × tokens_per_step × bytes × ep`; top-2 doubles it over top-1, so EP degree and interconnect (NVLink/IB) must be chosen against the measured communication share.
- **Utilization floor:** aim for 8–64 tokens per expert per GPU per micro-batch, i.e. `micro_batch * seq_len * 2 / (64/ep) ≥ 8`, and set gradient accumulation so `micro_batch * grad_accum * dp` hits the target global batch.

## Risks

- **Over-parameterization (highest):** 43.76B total vs 1.63B activated. Checkpoints and optimizer states are 27× the dense model's; the design is only viable if the cluster's sharded memory budget and checkpoint bandwidth accommodate it. Mitigate with EP + ZeRO-3 and state-sharded checkpoints.
- **All-to-all cost:** 64 experts × top-2 = large dispatch/receive volume per step; if interconnect is slow the run becomes communication-bound despite flat FLOPs. Mitigate with a higher EP degree, communication/compute overlap, and dropping the capacity factor to 1.0 once balanced.
- **Router collapse:** without the aux loss, the router would collapse onto a few experts and nullify the 64-expert investment. Mitigated by `aux_loss = 0.01`, jittered router logits, and capacity ≥ 1.0 so under-loaded experts keep gradients.
- **Padding / token drops:** at capacity factor 1.25 under imbalance, capacity slots are padded (wasted compute); lowering to 1.0 risks dropping tokens and distorting the effective batch. Monitor overflow fraction and balance before cutting capacity.

---

All figures follow the skill's formulas (`ffn_dim = ffn_mult*d_model`, `per-expert FFN (GLU) = 3*d_model*ffn_dim`, `embedding = vocab*d_model`); totals reconcile exactly with the dense, MoE, and activated parameter equations.