I'll start by exploring the workspace to find the skill resources (calculator, estimators, configs) referenced by these skills.
# MoE Throughput Optimization Report

**Assumed architecture** (shared 7B-MoE / ~1B dense-equivalent config from `moe-training`): 16 layers, `d_model=2048`, expert FFN `ffn_dim=1024`, 64 experts, top-2, `vocab=32768`, `seq_len=2048`, capacity factor 1.0. All figures below follow `tools/moe_calculator.py`, `tools/memory_estimator.py`, and `profilers/throughput_profiler.py` and can be re-derived by running them against this config.

**Baseline (given):** 12,400 tok/s, 62% GPU util, 8 GPUs, EP=8 (so DP=1, TP=1, PP=1), `micro_batch=4`, no activation checkpointing, FP32 optimizer states.
- Activated params = 16×(4·2048² + 2·2048 + 2·3·2048·1024) + 32768·2048 ≈ 0.54B
- FLOPs/token ≈ 6×0.54B + 4×16×2048×2048 ≈ 3.49 TFLOPs → MFU ≈ 17% (rough estimate; A100-80G peak assumed)

## Changes

Ranked by ROI:

1. **`micro_batch_size` 4 → 32, and enable activation checkpointing.**
   With EP=8 on 8 GPUs there is zero data parallelism (DP=1), so `micro_batch=4` *is* the whole per-step workload. An 8× larger micro-batch yields 8× larger GEMMs, fewer launches, and better comm/compute overlap — directly attacking the 38% idle fraction behind the 62% utilization. Checkpointing halves activations (5.4 → 2.7 GB at mb=4; holds the mb=32 footprint at 21.5 GB), keeping per-GPU memory ~30 GB. The +20–30% recompute cost is absorbed by the currently idle capacity. Lower `gradient_accumulation_steps` 8× to keep the global batch constant.
   **Verify:** A/B step-time run, mb=4 vs 32, fixed seed + global batch → expect +20–30% tok/s.

2. **Grouped matmuls over the 8 local experts** instead of a per-expert FFN loop.
   EP=8 shards 64 experts into 8 per rank; a sequential per-expert loop starves the GPU between 2048×1024 GEMMs. Batching the 8 local experts into one grouped GEMM is the largest single kernel win on the dominant compute region (expert FFN).
   **Verify:** benchmark grouped vs looped GEMM (Nsight) on the expert-FFN region → expect +10–30% there.

3. **Sequence packing** to fill capacity exactly instead of padding to the capacity factor.
   Padding wastes compute and adds bubble time; packing converts it to useful work.
   **Verify:** util with packing on vs off at fixed tokens/step → expect +10–15% tok/s.

4. **Conditional only — strengthen aux loss / add router jitter if router stats show imbalance.** No imbalance evidence was supplied, so do not change preemptively. Keep top-2; dropping to top-1 halves dispatch volume but risks quality.

## Before/After

| Metric | Before | After (projected) | Delta |
| --- | --- | --- | --- |
| Tokens/sec | 12,400 | ~17,500 | **+41%** |
| GPU utilization % | 62% | ~85% | +23 pts |
| Memory / GPU (GB) | ~12.5 | ~30.4 | +17.9 (still fits 40/80 GB) |

Memory math (per skill): params 0.77 + grads 0.77 + FP32 optimizer 3.09 + activations (5.37 at mb=4 no-ckpt → 21.5 at mb=32 with ckpt) + ~10% overhead + 1.5 GB buffers. MFU rises ~17% → ~24% (rough).

## Risks

- **Token drops / imbalance:** do not lower the capacity factor below 1.0; monitor overflow fraction and router entropy (`router_distribution.py`) after raising the micro-batch.
- **Checkpointing recompute (+20–30% compute):** safe while utilization is below 100%; if the batch change saturates the GPUs, re-balance recompute vs batch.
- **Packing complexity:** sequence packing adds attention-masking logic; a bug silently changes the effective batch — gate it behind an A/B.
- **Grouped-matmuls are framework/kernel-dependent:** the +10–30% band must be confirmed by benchmark, not assumed.
- **Projections are estimates:** confirm each gain with A/B runs (fixed seed, fixed global batch) before merging; the combined +41% is a compounding projection, not a guarantee.