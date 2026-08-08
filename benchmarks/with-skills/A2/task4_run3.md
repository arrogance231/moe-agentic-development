# MoE Throughput Optimization Report

**Assumptions** (state them so numbers are auditable): mid-scale MoE, top-2 routing, EP=8 on 8 GPUs (`DP=TP=PP=1`), BF16 compute with FP32 (AdamW-mixed) optimizer states, micro-batch 4, no activation checkpointing. Throughput analysis follows the moe-performance workflow; all gains are estimates pending an A/B profiler run.

## Changes

| # | Change | Why | Expected gain |
|---|---|---|---|
| 1 | **Enable gradient checkpointing** | Activations are the only per-GPU memory you can shrink without touching parameters/optimizer states; it buys the headroom that makes #2 safe. Cost: +20–30% recompute in backward. | Enables #2 |
| 2 | **Raise `micro_batch_size` 4 → 8** | Micro-batch 4 is the primary cause of the 62% util: GEMMs are small, per-GPU expert utilization is below the 8–64 tokens/expert floor, and padding overhead is a larger fraction of each step. Doubling the batch directly improves matmul shape and expert utilization. | +20–30% tokens/sec, util → ~78% |
| 3 | **Lower capacity factor 1.25 → 1.0** | With EP=8 every MoE layer runs an all-to-all; a high capacity factor inflates both dispatch volume and padding compute. Cut it **only after** experts are balanced (see #4). | +3–8% tokens/sec |
| 4 | **Raise aux loss 0.001 → 0.01** | Strengthens load balancing so the lower capacity factor doesn't start dropping tokens — it is the prerequisite for #3 and directly lifts expert utilization. | +5–10% expert util |
| 5 *(optional)* | **BF16/FP8 optimizer states** | FP32 Adam states dominate memory; reducing them frees ~5–15 GB/rank for an even larger batch. Precision risk — benchmark per workload. | Memory headroom |
| 6 *(optional)* | **Sequence packing / grouped matmuls** | Packing removes padding to capacity; grouped GEMMs replace per-expert loops. | +10–15% / +10–30% on expert FFN |

Order of operations: 1 → 2 first (biggest lever), verify no OOM, then 3 → 4 together, then decide on 5/6 from the profiler.

## Before/After Metrics

| Metric | Before | After | Delta |
|---|---|---|---|
| tokens/sec (global) | 12,400 | ~15,600 | **+26%** |
| GPU utilization (%) | 62% | ~78% | +16 pts |
| Memory (per GPU) | ~70 GB | ~72 GB | ~flat |

Memory is optimizer-state-dominated, so it barely moves: checkpointing halves activations (≈ −2.7 GB at batch 4) while the doubled micro-batch restores them (≈ +5.4 GB). The win is *utilization*, not footprint. With the optional BF16 optimizer states (#5), memory drops to ~55 GB and can fund a batch of 12–16.

## Risks

- **OOM at batch 8.** The activation estimate is rough; if the run dies, fall back to batch 6 rather than reverting checkpointing. Keep ≥20% headroom and confirm with the memory estimator before the full run.
- **Gradient-checkpointing recompute erodes gains.** If the run is compute-bound rather than memory-bound, the +20–30% backward cost can exceed the utilization win — A/B against checkpointing off at batch 8.
- **Token drops from cf 1.0.** If aux loss hasn't balanced experts, lowering capacity factor drops tokens and quietly changes the effective batch/quality. Monitor the overflow fraction; if nonzero, stay at 1.25.
- **Aux loss distortion.** 0.01 can start overriding the routing objective — watch router entropy and validation quality, and back off to 0.005 if quality dips.
- **Precision loss (optional #5).** FP8/BF16 optimizer states can destabilize training; only adopt after confirming loss stability over a few hundred steps.
- **Gains are estimates, not guarantees.** Total expected improvement is ~26% from realistic levers (single-digit to ~15% each, kernel-level up to ~40%) — not an order of magnitude. Verify each change with a fixed-seed A/B step-time run and `profilers/throughput_profiler.py` before committing.