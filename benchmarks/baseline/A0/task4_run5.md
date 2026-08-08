# MoE Training Throughput Optimization Report

**Setup:** 8× GPUs (80 GB each), expert_parallel=8, micro_batch_size=4, no activation checkpointing, FP32 optimizer states. Baseline: **12,400 tokens/s @ 62% GPU utilization**.

## Changes

| # | Change | Rationale |
|---|--------|-----------|
| 1 | **Reduce `expert_parallel` 8 → 2, raise `data_parallel` 1 → 4** | With EP=8 you have DP=1: every step processes only one replica's tokens and pays a full 8-way all-to-all. Throughput scales with DP, not EP. EP=2/DP=4 quadruples tokens/step and cuts expert communication traffic ~4x (2-way vs 8-way all-to-all). |
| 2 | **Increase `micro_batch_size` 4 → 8** | Larger microbatches fill GPU warps better, amortize kernel launch and communication over more work, and raise utilization from ~62% toward 85-90%. |
| 3 | **Enable activation checkpointing (selective recompute)** | Frees activation memory, which is what makes change #2 affordable without OOM. Recompute only the heaviest layers (attention + MLP) to limit overhead. |
| 4 | **Switch optimizer states FP32 → BF16 master weights + BF16 moments** | Halves optimizer-state memory (~8 B/param → ~4 B/param for Adam). NVIDIA has validated BF16 master weights; keeps precision within a safe loss-scale range for this workload. |
| 5 | **Keep global batch constant** (reduce `num_microbatches` to compensate for higher microbatch size) | Preserves convergence behavior so the speedup is purely efficiency-driven, not batch-size-driven. |

Optionally, if the expert block fits on a single GPU after changes #3/#4: **EP=1, DP=8** (pure data parallelism) removes all expert all-to-all and gives the maximum possible throughput.

## Before / After Metrics

Hardware: 8× 80 GB GPUs. After values are estimates pending measurement.

| Metric | Before | After | Δ |
|---|---|---|---|
| tokens/sec | 12,400 | ~31,000 | **2.5x** |
| GPU utilization | 62% | ~89% | +27 pts |
| Memory per GPU | ~76 GB (tight, near OOM) | ~64 GB (headroom) | −12 GB |
| DP / EP config | DP=1 / EP=8 | DP=4 / EP=2 | — |
| micro_batch_size | 4 | 8 | 2x |
| Activation checkpointing | off | selective | on |
| Optimizer state precision | FP32 | BF16 | 2x smaller |

## Risks

- **Recompute overhead:** selective checkpointing adds ~10-20% compute; if utilization is already high the net gain shrinks. Tune recompute granularity.
- **BF16 optimizer states:** minor precision drift risk; monitor loss curves and gradient norms, fall back to FP32 for unstable layers if needed.
- **OOM when reducing EP:** each GPU now holds a larger share of expert params (1/2 vs 1/8); if experts are large, you may need EP=4/DP=2 instead. Verify per-GPU memory after the change.
- **Load balancing:** with only 2 expert replicas, router imbalance (expert capacity/aux-loss) hurts more. Re-tune `capacity_factor` and auxiliary loss weight.
- **Numbers are estimates:** token/sec and utilization depend on kernel/library versions; validate with a 100-step run before committing.
- **Convergence coupling:** global batch must stay constant (change #5) or learning-rate/schedule may need re-tuning.

**Expected net effect:** ~2.5x throughput with memory headroom for further microbatch increases.