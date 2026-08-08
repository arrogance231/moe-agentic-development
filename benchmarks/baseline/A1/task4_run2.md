Let me check the working directory context first.
# MoE Training Throughput Optimization Report

**Baseline:** 12,400 tok/s, 62% GPU util, 8× GPUs, `expert_parallel=8`, `micro_batch_size=4`, no activation checkpointing, FP32 optimizer states.

## Changes

| # | Change | Why |
|---|--------|-----|
| 1 | Enable **selective activation checkpointing** (recompute only `layernorm`, `up_proj`, `gate`; keep attention/expert forward outputs) | MoE activations — routing buffers + SwiGLU intermediates — dominate memory. Full recompute would add ~33% FLOPs and re-trigger EP all-to-all in backward; selective recompute reclaims most memory at <5% overhead. |
| 2 | Raise `micro_batch_size` 4 → **8** (keep global batch via `gradient_accumulation`) | Larger, denser expert GEMMs and fewer kernel launches directly attack the low utilization: 62% util on 8 GPUs is the classic "skinny GEMM + host-bound" signature, not a saturation of the all-to-all link. |
| 3 | Switch **FP32 → BF16 (precision-aware) optimizer states**, keep FP32 master | Cuts optimizer memory 2× (m/v from 4B to 2B each) with negligible convergence impact for MoE — the freed HBM funds the larger micro-batch. |
| 4 | Overlap EP dispatch/combine all-to-all with expert compute (`--overlap-moe-expert-parallel-comm`) | Hides the EP communication wall behind GEMMs instead of serializing it; on 8 GPUs this is the next bottleneck after utilization. |
| 5 | Use **grouped GEMM** + router/permute fusion (`--moe-grouped-gemm`, `--moe-router-fusion`, `--moe-permute-fusion`) | Replaces many tiny per-expert kernels with one batched GEMM; directly recovers the SM efficiency loss in small expert matrices. |
| 6 | Enable **sequence parallelism** + CUDA Graphs | Shards layernorm/dropout activations and removes host-side kernel launch overhead — a primary driver of the sub-80% utilization ceiling. |
| 7 | Optional (Hopper+): **FP8 expert GEMMs** | Further memory + GEMM-throughput gain if convergence allows; treat as a phase-2 change since it interacts with change 1. |

**Order of operations matters:** enable checkpointing and BF16 optimizer states *first* to create memory headroom, then grow `micro_batch_size`, then overlap/fuse. Each step unblocks the next; doing fusion before there is enough batch to fuse gains little.

## Before / After Metrics

Values are projections for an 8× H100/H200-class node, to be validated with a profiling pass (nsys/nsight) after each change.

| Metric | Before | After | Expected Δ |
|--------|--------|-------|-----------|
| Throughput (tok/s) | 12,400 | ~20,000–22,000 | **+60–75%** |
| GPU utilization | 62% | ~80–85% | +18–23 pp |
| Peak mem / GPU (GB, 80 GB HBM) | ~72 (activation-bound) | ~62–68 (recomputed+BF16 states, offset by MBS=8) | freed, re-spent on batch |
| Optimizer-state mem / param | 12 B (fp32 m+v) | 8 B (bf16 m+v) | −33% |
| Micro-batch size | 4 | 8 | 2× GEMM size |

Largest single lever is the MBS bump (typically +35–45% from batch alone); overlap and fusion contribute the rest. Do not expect the memory column to drop after change 2 — the freed memory is deliberately reinvested in batch.

## Risks

- **Activation recompute cost:** if recompute scope is too aggressive (e.g., including attention or experts), it re-runs EP all-to-all in backward and can *lower* throughput. Recompute only cheap ops; profile after each scope change.
- **BF16 optimizer states:** minor convergence drift possible for very small gradients/EMAs; monitor loss curves and revert to FP32 if quality regresses.
- **Larger micro-batch changes routing dynamics** — token-drop/load-balance statistics shift, can raise all-to-all volume and worsen imbalance between experts.
- **Grouped GEMM + dropless routing** interplay: batching requires static-ish expert loads; high routing skew degrades the grouping benefit and can reintroduce stragglers.
- **CUDA Graphs** conflict with dynamic (dropless) token routing — may need a fixed-capacity router or per-shape graph capture.
- **FP8** adds quantization kernels and precision risk; only adopt after FP8/FP4 recipe validation, and never before the memory/batch/overlap changes are in place.
- Each fix can expose a new wall (memory → comm → host overhead); re-profile between changes rather than applying all at once.