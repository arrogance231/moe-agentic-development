# MoE Throughput Optimization Report

**Baseline:** 12,400 tokens/sec on 8 GPUs at 62% GPU util — `expert_parallel=8` (DP=1), `micro_batch_size=4`, no activation checkpointing, FP32 optimizer states.

## Changes

| # | Change | Why |
| --- | --- | --- |
| 1 | **`micro_batch_size` 4 → 8** | mb=4 is below the 8–64 tokens-per-expert-per-GPU floor; small batches under-fill the GPU and are the most likely driver of the 38% idle time. Doubling the batch raises compute density per step. Expected +10–15% util. |
| 2 | **Enable activation checkpointing** | With no checkpointing, activations are the memory ceiling that caps the micro-batch. Checkpointing halves activation memory and funds the batch increase. Costs +20–30% compute, but at 62% util there is headroom, so net positive. |
| 3 | **BF16 mixed precision (FP32 master weights)** | FP32 activations/params double matmul time and activation memory vs BF16. Compute in BF16, keep FP32 master: +20–40% on compute-bound regions and halves activation memory, further funding change #1. |
| 4 | **`expert_parallel` 8 → 4 (DP=2)** | EP=8 on 8 GPUs leaves no data parallelism and fires an all-to-all every MoE layer at maximum dispatch volume. EP=4 halves per-step all-to-all bytes; the extra per-rank expert memory (16 experts/rank vs 8) is affordable after #2 and #3 free memory. Expected +5–10% tokens/sec if communication-bound. |

Optional 5: **sequence packing** if profiler shows padding waste, instead of padding to the capacity factor (+10–15%).

## Before/After

| Metric | Baseline | Proposed | Delta |
| --- | --- | --- | --- |
| Tokens/sec | 12,400 | ~17,700 | **+43%** |
| GPU utilization | 62% | ~85% | +23 pts |
| Memory (per GPU) | fits at mb=4, FP32 acts, no ckpt | fits at mb=8, BF16 acts, ckpt on | ~neutral (±10%) |

Gains are estimates from the skill's standard bands, applied sequentially (checkpointing frees activation memory, BF16 halves it again, so the doubled batch lands at roughly baseline memory).

## Risks

1. **OOM at mb=8** — if activations still dominate after BF16 + checkpointing. *Mitigation:* validate with the memory estimator first; step down to mb=6 if it doesn't fit.
2. **Checkpointing compute overhead (+20–30%)** — erodes the net gain if the GPU is more compute-bound than utilization suggests. *Low risk* at 62% util; verify with an A/B step-time run.
3. **BF16 precision instability** — loss spikes/NaN or router drift. *Mitigation:* keep FP32 master weights, monitor loss; fall back to FP16 + loss scaling or FP32 compute if it diverges.
4. **EP=4 reduces load-balancing granularity** — 16 experts/rank can increase padding under expert skew and doubles per-rank expert memory. *Mitigation:* A/B step time at EP=8 vs EP=4; revert if step time doesn't drop.
5. **Combined-change attribution** — applying all four at once obscures what worked. *Mitigation:* apply in order (1→4), profiling tokens/sec and util after each change.