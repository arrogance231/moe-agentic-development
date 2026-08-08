# MoE Throughput Optimization Report

**Reference config assumed for memory math** (model size not given — adjust figures if yours differs): ~1B dense-equivalent, 64 experts, top-2, d_model=2048, 24 layers, vocab=32k, seq_len=2048, 8 GPUs, EP=8 (→ DP=1). All "After" throughput/util figures are *estimates* to be confirmed by A/B runs, not guarantees.

## Changes

1. **Run forward/backward in BF16, keep FP32 master weights (do NOT touch the FP32 optimizer states).**
   FP32 matmuls run off tensor cores; BF16 roughly doubles matmul throughput on A100/H100 and halves activation memory. FP32 optimizer states are the *correct* default for AdamW — the fix is that params/grads stop being FP32, not that the optimizer states do. Fall back to TF32 (near-FP32 quality, ~1.5–2× FP32 on A100) only if BF16 shows NaN. *Verify:* A/B step-time runs; watch loss curve for NaN/spikes.

2. **Enable gradient checkpointing.**
   No checkpointing + MoE capacity buffers = activations dominate memory (see table). Recomputation halves activation memory at +20–30% compute, buying headroom for Change 3. *Verify:* run `tools/memory_estimator.py` before/after; monitor `nvidia-smi` peak.

3. **Raise `micro_batch_size` 4 → 8 and halve `gradient_accumulation_steps` to hold the global batch constant.**
   At micro_batch=4 the GPUs are underutilized (62%). With DP=1, global_batch = micro_batch × grad_accum, so 4→8 with grad_accum halved keeps the batch identical while amortizing kernel-launch and all-to-all overheads. Tokens per expert stays far above the 8–64 floor (8 × 2048 × 2 / 8 ≈ 4096). *Verify:* sweep micro_batch 4/6/8 and measure tokens/sec + util per step.

4. **Conditional — strengthen the load-balancing aux loss (0.001 → 0.01).** Only if router stats show skew: with EP=8, imbalanced routing leaves expert capacity idle and pads to capacity. *Verify:* `analyzers/router_distribution.py` — confirm effective experts ≥ 0.5 × n before/after.

5. **Conditional — overlap all-to-all with compute.** EP=8 on 8 GPUs means one all-to-all per MoE layer with DP=1. If a step-time breakdown shows comm dominating after Changes 1–3, enable comm/compute overlap rather than cutting EP (EP=4 + DP=2 would shrink per-layer traffic but double experts per GPU). *Verify:* compare step time at EP=8 vs EP=4 on identical config.

## Before/After Metrics

| Metric | Before | After (est.) | Delta |
| --- | --- | --- | --- |
| Tokens/sec | 12,400 | ~17,400 | **+40%** |
| GPU utilization | 62% | ~80% | +18 pts |
| Peak activation memory/GPU* | ~32 GB (FP32, no ckpt, mb=4) | ~16 GB (BF16 + ckpt, mb=8) | **−50%** |
| Param + optimizer footprint/GPU | baseline | ~unchanged (FP32 master kept) | ~0% |

\*Rough estimate via the `×20`-factor activation formula: `num_layers × micro_batch × seq_len × d_model × 20`. The freed ~16 GB/GPU is what funds the micro_batch increase.

## Risks

1. **BF16 precision loss / NaN** — mitigated by FP32 master weights and gradient clipping (norm 1.0); TF32 as fallback.
2. **Checkpointing can slow things if memory wasn't binding** — it only pays off combined with Change 3; revert to selective checkpointing if step time regresses on its own.
3. **OOM on the micro_batch bump** — the activation estimate is rough and MoE capacity buffers inflate it; ramp 4 → 6 → 8 stepwise under `nvidia-smi` watch.
4. **Communication cap at EP=8** — all-to-all volume scales with tokens/step, so the mb increase adds dispatch traffic; could limit the gain to below +40%.
5. **Aux-loss over-strengthening distorts routing** — cap at 0.01 and gate Change 4 on measured skew.
6. **All deltas are estimates** — confirm each change independently (fixed seed/data) before stacking, or an interaction may eat a nominal gain.