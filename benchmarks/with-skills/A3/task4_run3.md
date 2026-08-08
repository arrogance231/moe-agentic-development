I'll follow the `moe-performance` procedure (baseline → bottleneck analysis → ROI-ranked changes → metrics → risks). No architecture was supplied, so I estimate memory/MFU against the skill's reference config (7B-MoE / ~1B dense-equivalent, 64 experts, top-2, 32 layers, d=4096, seq=2048, 8×80GB GPUs) and flag that as an assumption.

# MoE Training Throughput Optimization Report

## Diagnosis (baseline)

| Metric | Value | Reading |
| --- | --- | --- |
| tokens/sec | 12,400 | Low for 8 GPUs at this scale |
| GPU util | 62% | 38% bubble/idle per step |
| EP | 8 = all 8 GPUs | One all-to-all per MoE layer; every token crosses the interconnect |
| micro_batch | 4 | Small matmuls, high relative padding |
| optimizer | FP32 states | ~12 B/param for AdamW; largest single memory line item |
| est. MFU | ~4% | Device busy 62% but few useful FLOPs → communication + padding + small-kernel bound, not compute bound |

**Bottleneck ranking (per `tools/bottleneck_rank.py` logic):** (1) low GPU util from small micro-batch and padding, (2) all-to-all volume with EP=8 and no comm/compute overlap, (3) memory pressure from FP32 optimizer states + no checkpointing that blocks batch growth.

## Changes

### Change 1 — Enable gradient checkpointing and raise `micro_batch_size` 4 → 8
**Why.** 62% util with `micro_batch_size=4` is the classic small-batch/under-utilization signature: expert matmuls are small and padding-to-capacity waste grows. Doubling the batch raises matmul efficiency and amortizes fixed per-step overhead. Doubling activations (~21 GB → ~43 GB at seq 2048) would OOM on 80 GB GPUs, so checkpointing (~halves activation memory, +20–30% recompute) makes the batch increase fit at roughly unchanged peak memory.
**Expected gain.** ~+17% tokens/sec (→ ~14,500), util → ~75%. Biggest ROI of the three.
**Verify.** A/B `profilers/throughput_profiler.py` at mb=4 vs mb=8 with checkpointing on; confirm no OOM and util delta ≥10 pts.

### Change 2 — Cut all-to-all cost: enable comm/compute overlap, `capacity_factor` 1.25 → 1.0, `router_aux_loss_coef` 0.01
**Why.** EP=8 means all-to-all every MoE layer is mandatory; you cannot raise EP further. The levers left are dispatch volume and overlap. A cf of 1.0 (instead of 1.25) shrinks padding and dispatch volume; an aux loss at 0.01 keeps load balanced so lowering cf doesn't drop tokens. Overlapping the all-to-all with matmul compute hides the interconnect wait that drives part of the 38% bubble.
**Expected gain.** ~+5–10% tokens/sec (→ ~15,300), util → ~78%.
**Verify.** Compare step-time share of communication with the framework's overlap flag on/off; monitor overflow fraction in `analyzers/router_distribution.py` (must stay ~0 at cf=1.0).

### Change 3 — FP32 optimizer states → BF16 optimizer states (keep FP32 master weights)
**Why.** AdamW FP32 states are ~12 B/param and are memory-bandwidth-bound to update. BF16 moments cut this to ~8 B/param, freeing ~7 GB/GPU and shortening the optimizer step, while the FP32 master preserves quality.
**Expected gain.** ~+2–5% tokens/sec (→ ~15,600) and ~−10 GB/GPU headroom.
**Verify.** Same-seed 100-step run comparing loss curves; confirm no drift and rerun the memory estimator.

### Change 4 (secondary, only if util plateaus <80%) — Sequence packing
Pack sequences to fill expert capacity exactly instead of padding to the capacity factor. **~+10–15% tokens/sec**, risk: attention-masking complexity.

## Before/After metrics

| Metric | Before | After | Delta |
| --- | --- | --- | --- |
| tokens/sec (global) | 12,400 | ~15,600 | +26% |
| GPU utilization (%) | 62 | ~79 | +17 pts |
| Peak memory / GPU* | ~56 GB | ~48 GB | −14% |

*Estimates from `tools/memory_estimator.py` on the reference 7B-MoE/64-expert/top-2/seq-2048 architecture (weights 3.5 + grads 3.5 + optimizer 21 + activations 21.5 + ~10% overhead). Checkpointing + BF16 states offset the 2× micro-batch: activations stay ~21 GB, optimizer drops to ~14 GB.

Per-change cumulative: 12,400 → 14,500 (C1) → 15,300 (C2) → 15,600 (C3).

## Risks

- **Token drops at cf=1.0** — if residual expert skew exceeds capacity, tokens drop and the effective batch shrinks. Mitigate: aux loss 0.01, keep cf ≥1.0, watch overflow fraction.
- **Checkpointing compute penalty** — +20–30% recompute could outweigh the batch gain if the run is more compute- than communication-bound. Mitigate: A/B before committing.
- **OOM on the larger micro-batch** — on 40 GB GPUs (not the assumed 80 GB), mb=8 may still exceed. Mitigate: step down to mb=6, confirm with the memory estimator.
- **BF16 optimizer precision** — low-magnitude second-moment loss can drift quality. Mitigate: keep FP32 master weights, monitor the loss curve on the fixed-seed ablation.
- **Aux loss distortion** — 0.01 is the top of the safe band; overshooting it biases the router away from the task loss.
- **Overlap misconfiguration** — framework-specific flags can silently add traffic rather than hide it; verify the comm share of step time drops.