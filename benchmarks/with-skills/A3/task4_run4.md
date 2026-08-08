Note: I'll assume a representative config since the full spec wasn't given — 64 experts, top-2, seq_len 2048, ~1B-dense-equivalent MoE (~3B total params), 8× A100-80GB, EP=8 ⇒ DP=1. Every number below should be re-validated with `profilers/throughput_profiler.py` and `tools/bottleneck_rank.py`.

# MoE Training Throughput Optimization Report

**Symptom:** 12,400 tok/s at 62% GPU util on 8 GPUs (EP=8, micro_batch=4, no checkpointing, FP32 optimizer states). At 62% util the GPUs are idle ~38% of the step — the dominant lever is filling that idle time, not reducing FLOPs.

## Changes

1. **Enable activation checkpointing** (no config change needed to memory targets, big unlock). Activations at micro_batch=4/seq 2048 are ~16 GB/rank of a ~40 GB footprint. Recompute roughly halves activation memory at +20–30% compute — but on a 62%-util run that extra compute lands in existing idle cycles, so it is largely free. *Why:* removes the memory ceiling that blocks change #2.
2. **Double the micro-batch: 4 → 8**, and halve gradient-accumulation steps to keep the global batch identical. *Why:* the run is utilization-bound, not quality-bound; micro_batch=4 underfills the GPU. With checkpointing (#1) and BF16 optimizer states (#3), 8 fits comfortably. Expected +25–35% tokens/sec. *Verify:* A/B step-time run at micro_batch 4 vs 8, same seed; confirm util rises and OOM stays clear.
3. **Switch optimizer states from FP32 to BF16** (keep FP32 master weights optional). Halves optimizer-state memory from 12 → 6 bytes/param for the ~1B dense parameters replicated on every rank (DP=1, so they cannot be ZeRO-sharded). Frees ~6 GB/rank, widening headroom for #2. *Verify:* 100 steps with FP32 vs BF16 states; confirm loss tracks within noise.
4. **Grouped matmuls for expert FFNs.** With EP=8 each rank owns 8 experts; looping experts one-at-a-time is common at this layout. Batch expert inputs into grouped GEMMs. Expected +10–30% on expert-FFN time. *Verify:* benchmark per-expert loop vs grouped GEMM on the same tensors.
5. **Sequence packing.** If the dataloader pads to seq_len, packing recovers +10–15% tokens/sec. *Why:* padding to capacity wastes the compute #2 paid for. *Verify:* tokens/sec with packing on vs off at fixed global batch.
6. **Only after #1–5:** inspect router distribution (analyzers/router_distribution.py). If skew or overflow exists, strengthen aux loss (0.001 → 0.01) or raise capacity factor; if balanced, leave untouched.

## Before/After Metrics

Assumes the stated config; re-run `profilers/throughput_profiler.py` to confirm.

| Metric | Before | After | Delta |
| --- | --- | --- | --- |
| tokens/sec (global) | 12,400 | ~17,500–18,500 | +40–50% |
| GPU utilization (%) | 62 | ~85–90 | +23–28 pts |
| Memory/GPU | ~40 GB | ~36 GB | −10% |
| micro-batch size | 4 | 8 | 2× |
| activation checkpointing | off | on | — |
| optimizer states | FP32 (12 B/param) | BF16 (6 B/param) | −6 GB/rank |
| gradient accumulation | — | halved | keeps global batch |

Memory per GPU *drops* despite the 2× micro-batch because checkpointing halves activations and BF16 optimizer states halve the replicated dense optimizer block. Changes #4/#5 add a further single-digit-to-15% on top; the +40–50% headline assumes only #1–#3.

## Risks

- **Micro-batch doubling → OOM** if the activation estimate is optimistic. Mitigate: enable #1 and #3 *first*, then raise micro-batch one step at a time; roll back to 4 if the 80 GB budget is exceeded.
- **BF16 optimizer states degrade convergence** (momentum/variance precision). Guard: keep FP32 master weights, run 100-step parity check; if loss diverges, revert and rely on #1/#2 alone.
- **Checkpointing adds +20–30% compute.** On a 62%-util run this hides in idle cycles, but if #2 raises util toward 90%+ first, the recompute overhead becomes visible. Order matters: enable checkpointing before the batch bump.
- **Grouped matmuls may not help on this framework/hardware** (kernel-dependent); benchmark before adopting — worst case neutral.
- **Sequence packing adds attention-masking complexity** and can subtly change the effective batch composition; validate loss parity on a short run.
- **Global-batch drift.** Halving grad_accum must exactly preserve `micro_batch × grad_accum × dp`; a mismatch changes the effective batch and LR schedule.