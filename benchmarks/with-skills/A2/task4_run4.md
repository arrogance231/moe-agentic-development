# MoE Throughput Optimization Report

**Target run:** 12,400 tok/s · 8 GPUs · EP=8 (DP=1) · micro_batch=4 · no checkpointing · FP32 optimizer states

**Assumptions:** ~7B-MoE / ~1B dense-equivalent, 64 experts, top-2, 24 layers, d_model 2048, seq_len 2048, bf16-capable GPUs (A100/H100), NVLink intra-node. Memory figures are hand-estimates from the moe-training/moe-performance formulas — re-run `tools/memory_estimator.py` on the real config before launch.

## Changes

**1. Enable gradient checkpointing and raise micro_batch_size 4 → 8 (halve grad_accum).**
mbs=4 is the direct cause of 62% utilization: matmuls are too small to fill the GPU and the EP all-to-all latency amortizes over too few tokens. Doubling the micro-batch raises per-kernel efficiency. Checkpointing frees ~50% of activation memory so mbs=8 fits; keep the global batch fixed by halving `grad_accum`. Verify with an A/B sweep mbs 4→6→8 at a fixed seed/steps. Expected +10–15% net after the ~+20% recompute cost is absorbed by the util gain.

**2. Enable communication/computation overlap for the EP all-to-all.**
EP=8 on 8 GPUs means DP=1 and a full 8-rank all-to-all per MoE layer (volume = top-2 × tokens/step per layer). Overlap hides dispatch/sum behind FFN compute — this is most of the remaining idle time. Verify with a step-time breakdown. Expected +5–8%.

**3. Confirm BF16 compute; keep FP32 optimizer states as-is.**
BF16 halves weight/grad memory and doubles tensor-core matmul throughput (+20–40% on compute-bound regions). If compute is already BF16, this is a no-op — FP32 optimizer states are standard and must stay (moving them to FP16 risks divergence and saves little). Make this change #1 only if the run currently computes in FP32.

**4. (Evidence-gated) Check router balance before touching capacity factor or aux loss.**
cf/aux changes carry quality risk and only pay if imbalance is actually padding experts. Run `router_distribution.py` / `expert_utilization.py` first; only then tune cf 1.25→1.0 or aux 0.001→0.01.

## Before/After

| Metric | Before | After | Delta |
| --- | --- | --- | --- |
| tokens/sec (global) | 12,400 | ~15,900 | +28% |
| GPU utilization | 62% | ~80% | +18 pts |
| Peak memory/GPU | ~36 GB | ~27 GB | −25% |

Memory driver: weights+grads drop 14 → 7 GB under BF16; optimizer states stay 14 GB (FP32, unchanged); activations fall from ~16 GB (FP32, un-checkpointed, mbs 4) to ~4–8 GB (BF16 + checkpointed at mbs 8). If compute is already BF16, expect the tokens/sec delta closer to +15–20%.

## Risks

- **OOM at mbs=8** — mitigated by checkpointing; de-risk with the 4→6→8 sweep, fall back to mbs 6.
- **Checkpointing recompute (+20–30% compute) may exceed the util gain** if memory was never the binding constraint — the A/B proves it; if net-negative, keep mbs=8 only if it fits un-checkpointed.
- **Global batch drift** — halving grad_accum when doubling mbs is mandatory, or the LR schedule silently mismatches the doubled batch.
- **Quality from capacity/aux tuning** — excluded from the core plan; pursue only with router evidence, inside cf 1.0–1.25 and aux 0.001–0.01.
- **Overlap correctness** — dispatch must keep identical token counts; verify pre/post-overlap counts to catch silent drops.