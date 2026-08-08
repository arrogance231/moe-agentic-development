# MoE Training Throughput Optimization Report

**Baseline:** 12,400 tok/s across 8x GPUs, 62% GPU utilization, `expert_parallel=8`, `micro_batch_size=4`, no activation checkpointing, FP32 optimizer states.

## Changes

**1. Enable selective activation checkpointing (recompute attention + MLP activations).**
Currently activations are held for all layers, so memory pressure caps the microbatch. Selective recomputation trades a small amount of compute for a large reduction in peak activation memory, which is the lever for everything below.

**2. Switch optimizer states from FP32 to BF16 (with FP32 master weights).**
Adam keeps 2 states/param; FP32 states = 8 bytes/param, BF16 = 4 bytes/param (+4 bytes FP32 master). Cuts optimizer memory ~30%, directly reducing the per-GPU memory floor with negligible accuracy impact when master weights stay FP32.

**3. Shard optimizer states across the 8 GPUs (`--use-distributed-optimizer` / ZeRO-1).**
EP=8 already shards the experts; sharding optimizer states reduces the optimizer footprint another ~8x per GPU. This is what actually breaks the memory ceiling.

**4. Increase `micro_batch_size` from 4 → 16 (via gradient accumulation for the same global batch).**
At 62% utilization the run is under-filled: small microbatches leave compute bubbles and underlap the expert all-to-all. The memory freed by changes 1–3 funds a 4x larger microbatch, which fills the pipeline, amortizes the all-to-all cost over more tokens, and improves compute/communication overlap.

**5. Overlap expert all-to-all with attention compute.**
Dispatch/all-gather for EP is the dominant bubble source at EP=8. Enable async dispatch (e.g. `--overlap-all2all` / DeepSpeed's EP comm overlap) so token dispatch runs concurrently with preceding attention, converting bubble time into useful FLOPs.

**6. (Optional) FP8 for expert GEMMs** if the hardware supports it — experts dominate FLOPs in MoE and get the largest win from FP8, though this carries accuracy risk (see below).

## Before / After Metrics

Estimates assume ~80 GB/GPU and a representative MoE; actual values scale with model size.

| Metric | Before | After (est.) | Δ |
|---|---|---|---|
| Throughput (tok/s) | 12,400 | ~22,000–24,000 | **+75–95%** |
| GPU utilization | 62% | ~87–90% | +25–28 pts |
| Peak memory/GPU | ~76 GB (activation-bound) | ~52 GB (checkpointing + sharded BF16 opt) | **−32%** |
| Micro-batch size | 4 | 16 | 4x |
| Effective FLOPS utilization | ~35% | ~60% | +25 pts |

Memory freed (~24 GB/GPU) goes back into a larger microbatch, which is why the utilization gain compounds with the checkpointing/optimizer savings.

## Risks

- **Recompute cost:** If the run is compute-bound rather than memory-bound, checkpointing adds ~15–30% recompute FLOPs and can *lower* throughput. Validate the memory floor first before enabling full recompute (start with selective).
- **BF16 optimizer states:** Reduces numerical fidelity of Adam's moments; can slow or destabilize convergence on small-momentum, low-precision-sensitive models. Keep FP32 master weights and monitor loss/validation curves early.
- **Sharded optimizer:** Slightly more communication per step (reduce-scatter/update/gather); on slower interconnects it can eat some of the memory win.
- **Bigger microbatch / global batch:** Changes the effective batch size and gradient noise; may require LR-warmup or schedule retuning to preserve convergence quality.
- **All-to-all overlap:** Relies on large token buffers; if dispatch buffers are too small, overlap stalls and gains shrink. Also assumes NVLink-speed all-to-all — on Ethernet-based EP the comm can become the new bottleneck.
- **FP8 experts:** Accuracy risk, especially for the router/gating and low-precision-sensitive layers; require loss-scaling and careful calibration, and it only helps if GEMMs — not comm — are the real bottleneck.

**Recommendation order:** BF16/sharded optimizer → selective checkpointing → microbatch 4→16 → all-to-all overlap; measure after each step, since 62% utilization suggests communication overlap (change 5) likely contributes the single largest jump.