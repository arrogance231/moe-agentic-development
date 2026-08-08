# MoE Training Throughput Optimization Report

**Context:** 8× H100 (80GB), 12,400 tokens/s, 62% GPU utilization, `expert_parallel=8`, `micro_batch_size=4`, no activation checkpointing, FP32 optimizer states.

## Changes

| # | Change | Why |
|---|--------|-----|
| 1 | **Switch to BF16 mixed precision** (autocast + FP32 master weights) | Full FP32 compute does not use H100 tensor cores efficiently and doubles memory traffic. BF16 roughly halves optimizer/activation memory and roughly **doubles arithmetic throughput**. FP32 master weights preserve training quality. This is the single biggest lever. |
| 2 | **Enable selective activation checkpointing** (checkpoint attention, not MoE layers) | Frees activation memory, letting the batch size grow. Recompute only the cheap layers so FLOP overhead stays ~5–10%. |
| 3 | **Raise `micro_batch_size` 4 → 8** (freed by #1/#2), keep global batch via gradient accumulation | Bigger microbatch amortizes per-step overhead and improves tensor-core utilization — the current 62% util indicates the GPU is idle while waiting on communication/launch. |
| 4 | **Reduce EP 8 → 4, add `tensor_parallel=2`** (requires sequence parallelism per Megatron) | With EP=8 every MoE layer does a full 8-rank all-to-all. EP=4 + TP=2 halves the all-to-all rank count/volume per expert (NVLink-local), cutting dispatch latency — the main bottleneck in small-EP runs. |
| 5 | **Overlap all-to-all with compute** (Megatron async dispatcher / DeepEP / Hybrid-EP-style pipelined dispatch+compute+combine) | Dispatch sits on the critical path; overlapping it with the previous layer's expert GEMMs hides nearly all communication latency. NVIDIA Hybrid-EP reports up to 514% gains on all-to-all-bound MoE at scale. |
| 6 | **Enable fused kernels**: fused AdamW, fused router/load-balancing, GroupedGEMM for local experts (BF16) | Removes launch overhead and kernel gaps; GroupedGEMM cuts the per-expert GEMM turnaround that dominates MoE step time when experts per rank > 1. |

## Before / After Metrics

| Metric | Before | After (projected) |
|--------|--------|-------------------|
| Throughput (tokens/sec) | 12,400 | **~19,500 (+57%)** |
| GPU utilization | 62% | **~88%** |
| Peak memory / GPU | ~70 GB (batch-limited) | ~68 GB at 2× microbatch (BF16 + ckpt freed memory, consumed by larger batch) |

Projection basis: ~2× arithmetic gain from BF16 (compute becomes sub-dominant), plus ~25–35% from larger batch and hidden all-to-all, partially offset by recompute and TP=2 reduction in DP parallelism. Validate with `nsys`/`dcgm` profiles before committing.

## Risks

- **Numerical drift / loss spikes:** BF16 has lower dynamic range than FP32. Mitigate by keeping FP32 master weights, monitoring loss/valid, and spot-checking divergence vs. the FP32 baseline.
- **Recompute overhead:** Aggressive activation checkpointing adds FLOPs; if the run becomes compute-bound (not the case at 62% util), throughput can drop. Use selective checkpointing.
- **EP reduction changes communication topology:** TP=2 adds all-reduce per attention layer and shrinks data-parallel degree from 8 to 4 (per-group EP=4), so expert load balance must be re-verified; drop **revertible** to EP=8 if all-to-all was not actually the bottleneck.
- **Convergence schedule:** Doubling effective batch via gradient accumulation needs LR / warmup / batch-size scaling care.
- **Precision of projections:** Actual gains depend on the specific model (expert count, top-k, hidden size) and NVLink vs. InfiniBand fabric; the numbers above are estimates to be confirmed by profiling.