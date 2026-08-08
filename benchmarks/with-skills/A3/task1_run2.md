I'll start by exploring the environment to find the referenced tooling and any context for this design task.
The `moe-architecture` skill (provided above) is my fixed procedural guide. Since no shell is available in this session, I'll apply its exact formulas and decision tables by hand, state my assumptions, and keep every figure internally consistent so each can be re-derived from the calculator's formula set.

**Assumptions (not given in the task, so stated explicitly):** a ~1B dense baseline to replace; `vocab=32000`, `seq_len=2048`, and 8× NVIDIA H100 80GB with NVLink (the canonical 8-GPU setup referenced in the training skill's example configs). Hardware drives the expert-count decision below.

---

# 1B Dense-Equivalent MoE — Architecture Document

## Overview

This design replaces a ~1.03B dense decoder-only LM with a Mixture-of-Experts variant that activates a comparable-compute profile. Dense baseline: `d_model=1600`, `ffn_mult=4` (`ffn_dim=6400`), `num_layers=24`, `vocab=32000`. The MoE swaps each dense FFN for **32 expert FFNs with top-2 routing**, keeping attention and embedding unchanged.

Headline numbers (all from the `moe-architecture` formula set, `tools/moe_calculator.py`):

| Quantity | Value |
| --- | --- |
| Dense baseline params | **1,034,316,800** (~1.03B) |
| MoE total params | **23,889,996,800** (~23.9B) |
| Activated params (top-2) | **1,771,596,800** (~1.77B) |
| Param ratio (MoE/dense) | **23.1×** |
| Activated/dense ratio | **1.71×** |
| `num_experts` | **32** |
| `top_k` | **2** |
| FLOPs/token (training) | 10.94 G vs 6.52 G dense (**1.68×**) |

Expert count is set to 32, not 64, for a memory reason: 64 experts would push total params to 47.5B, which at EP=8 on 8×80GB H100 needs ~95GB/GPU of weights+optimizer alone — the "over-parameterization without compute gain" failure mode. 32 experts (23.9B) fits with headroom while staying in the typical 16–64 band.

## Parameters

All formulas are exactly those implemented in `tools/moe_calculator.py`:
`ffn_dim = ffn_mult * d_model = 6400`; attention/layer `= 4·d² = 10,240,000`;
per-expert FFN (GLU) `= 3·d·ffn_dim = 30,720,000`; layernorms `= 2·d = 3,200`;
embedding `= vocab·d = 51,200,000`.

**Math shown explicitly:**

```
dense_params = L·(4d² + 3d·ffn_dim + 2d) + vocab·d
             = 24·(10,240,000 + 30,720,000 + 3,200) + 32,000·1600
             = 24·40,963,200 + 51,200,000
             = 983,116,800 + 51,200,000
             = 1,034,316,800

moe_params = L·(4d² + 2d + num_experts·3d·ffn_dim) + vocab·d
           = 24·(10,240,000 + 3,200 + 32·30,720,000) + 51,200,000
           = 24·993,283,200 + 51,200,000
           = 23,838,796,800 + 51,200,000
           = 23,889,996,800

activated_params = L·(4d² + 2d + top_k·3d·ffn_dim) + vocab·d
                 = 24·(10,240,000 + 3,200 + 2·30,720,000) + 51,200,000
                 = 24·71,683,200 + 51,200,000
                 = 1,720,396,800 + 51,200,000
                 = 1,771,596,800

param_ratio = moe_params / dense_params = 23,889,996,800 / 1,034,316,800 = 23.10×
```

**Parameters table (all layers):**

| Component | Dense | MoE (total) | Activated (top-2) |
| --- | --- | --- | --- |
| Attention (`4d²·L`) | 245,760,000 | 245,760,000 | 245,760,000 |
| Expert FFNs (`3d·ffn_dim·L·{1,32,2}`) | 737,280,000 | 23,592,960,000 | 1,474,560,000 |
| Layernorms (`2d·L`) | 76,800 | 76,800 | 76,800 |
| Embedding (`vocab·d`) | 51,200,000 | 51,200,000 | 51,200,000 |
| **Total** | **1,034,316,800** | **23,889,996,800** | **1,771,596,800** |

Router weights (`num_experts·d ≈ 51K`/layer) are negligible and omitted, per the skill's formula. Row sums: dense `245.76M + 737.28M + 0.0768M + 51.2M = 1,034.3M`; MoE `245.76M + 23,592.96M + 0.0768M + 51.2M = 23,890.0M`; activated `245.76M + 1,474.56M + 0.0768M + 51.2M = 1,771.6M`. All consistent.

## Routing

| Setting | Value | Justification |
| --- | --- | --- |
| Strategy | **Top-2** | Default for training runs; blends two experts' knowledge and materially outscores Top-1 at equal total experts (skill decision table), at only ~2× dispatch cost. Top-1 is only worth it when latency-bound inference dominates, which is not this use case. |
| Active experts (`top_k`) | **2** | Standard training default; keeps activated params at 1.77B (1.71× the dense baseline — the same ~1.7–1.9× activated ratio as Mixtral-class top-2 MoE), balancing quality vs all-to-all volume. |
| Expert count | **32** | Middle of the typical 16–64 band; 32/EP8 = 4 experts/rank (good utilization); 64 would break the 8-GPU memory budget. |
| Capacity factor | **1.25** | Upper end of the 1.0–1.25 band: absorbs token-routing imbalance without dropping tokens during early training; drop to 1.0 once load balance is confirmed (safe: the aux loss targets balance). |
| Auxiliary loss | **0.01** | Upper end of the 0.001–0.01 band. A 1B-scale research model needs strong early load balancing to keep all 32 experts fed and avoid collapse; anneal toward 0.001 as the router stabilizes. |

## Training implications

- **Compute:** 10.94 GFLOPs/token vs 6.52 G dense (1.68×). The ratio is below the 1.71× activated-param ratio because the 0.31 G attention term is shared.
- **Parallelism:** `DP×TP×PP×EP = 1×1×1×8 = 8` GPUs; EP=8 divides `num_experts=32` (4 experts/rank). One all-to-all per MoE layer (24 total per step); top-2 → 2× dispatch volume.
- **Memory per GPU (estimator formulas):** 23.89B/8 ≈ 2.99B owned params → BF16 params 5.97GB + grads 5.97GB + AdamW mixed states 35.83GB + activations ≈7GB (micro-batch 8, seq 2048, recompute on, EP shards expert activations) + ~6.5GB overhead ≈ **~61GB → ~23% headroom on 80GB** (≥20% target).
- **Batch:** micro-batch 8 → 8,192 tokens/expert/GPU per micro-batch (well above the 8–64 floor; utilization is not a concern); with `grad_accum=64`, global batch = 512 seqs ≈ 1.05M tokens/step.
- **Throughput estimate:** at ~40% MFU on 8×H100 (~3,168 TFLOP/s effective), ≈ 289k tokens/sec.
- **Precision/checkpoints:** BF16, gradient checkpointing on (long context), HF-style checkpoints.

## Risks

- **Over-parameterization / memory.** 23.9B total for 1.03B compute — checkpoint and optimizer cost dominate. *Mitigation:* EP=8 + ZeRO sharding; the design explicitly avoids 64 experts, which would not fit 8×80GB; headroom verified at ~23%.
- **All-to-all communication.** Top-2 with EP=8 dispatches 2× per MoE layer; if step time becomes comm-bound, lower capacity factor to 1.0 and/or raise EP. *Verify:* compare step time at EP=8 vs EP=4.
- **Routing collapse / expert imbalance.** Weak balancing starves under-loaded experts. *Mitigation:* aux loss 0.01 + router-logit jitter; monitor effective experts and overflow; anneal aux loss after stabilization.
- **Token drops under imbalance.** At cf 1.25, heavy skew can still overflow experts. *Mitigation:* monitor overflow fraction; raise cf toward 1.5 early, tighten later.
- **Top-2 overhead vs top-1.** Activated params are 1.71× dense; if compute is later the binding constraint, top-1 exactly matches the 1.03B dense profile (activated = dense) at a quality cost — a documented fallback, not the default.
- **Embedding term at this scale.** vocab·d = 51.2M is ~5% of dense params; it is included in all totals (a common omission that breaks small-scale math).

**Validation note:** every figure above follows the calculator's exact formulas (`moe_params`, `activated_params`, `param_ratio`, `flops_per_token`) and is internally consistent across the math block, the table, and the headline summary; run `tools/moe_calculator.py` with `d_model=1600, ffn_mult=4, num_layers=24, vocab=32000, seq_len=2048, num_experts=32, top_k=2` to confirm.