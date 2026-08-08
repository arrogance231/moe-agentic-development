# MoE Training Throughput Optimization Report

## Baseline
- **Hardware:** 8x GPUs · **Throughput:** 12,400 tokens/s · **GPU util:** 62%
- **Config:** `expert_parallel=8`, `micro_batch_size=4`, no activation checkpointing, FP32 optimizer states

## Changes

**1. Replace pure EP with hybrid parallelism — EP=2, DP=4 (alt: TP=2, EP=4).**
With EP=8 on 8 GPUs there is zero data parallelism: every token crosses an 8-way all-to-all at every MoE layer, and each GPU's expert GEMMs are small (1/8 of experts). All-to-all is latency/bandwidth-bound and is the main reason utilization sits at 62%. Shrinking the EP group to 2 cuts dispatch/combine volume ~4×, makes expert GEMMs ~4× larger (better SM saturation), and turns gradient sync into an efficient 4-way reduce-scatter. Cost: experts are replicated 4×, so parameter/optimizer memory per GPU rises.

**2. Enable selective activation checkpointing.**
With no checkpointing, activations are all live in memory, which caps `micro_batch_size=4`. Selective recompute (attention/expert blocks only, or full-layer) frees gigabytes per GPU and is the enabler for #3. Expect ~10–20% extra FLOPs — worthwhile here because the run is memory/comm-bound, not compute-bound.

**3. Raise `micro_batch_size` 4 → 8 with gradient accumulation to keep global batch constant.**
Larger microbatches amortize kernel-launch and all-to-all latency, pushing utilization toward saturation. Pair with grad-accum (×2) so the effective update stays identical.

**4. Shrink optimizer state: FP32 → BF16 moments, keep FP32 master weights (or ZeRO-1/2 + offload).**
FP32 Adam states cost ~16 B/param. BF16 first/second moments with an FP32 master copy cut optimizer memory ~30–40%. This is what funds the increased memory from #1 and #3.

**5. Enable sequence parallelism + FlashAttention (BF16).**
Seq-parallel shards layernorm/dropout/embedding activations; FlashAttention drops attention memory from O(seq²) to O(seq). Both lower peak activation memory and raise achievable batch.

**6. Overlap all-to-all with compute.**
Run expert dispatch/combine on a separate stream overlapped with the previous layer's FFN, and overlap gradient reduce-scatter with backward (Megatron `--overlap-p2p-comm`/expert-comm overlap, DeepSpeed `all_to_all_overlap`). Hides the residual communication.

**7. (Optional) Fused kernels + graph capture.**
Grouped GEMM for experts, end-to-end BF16, CUDA graphs / `torch.compile` to eliminate launch overhead once communication is no longer the bottleneck.

## Before/After Metrics

| Metric | Before | After (target) | Driver |
|---|---|---|---|
| Throughput (tokens/sec) | 12,400 | ~23,000–27,000 | #1, #3, #5, #6 |
| GPU utilization | 62% | ~85–90% | #1, #3, #6 |
| Peak activation memory/GPU | high (no ckpt, no seq-par) | −40–60% | #2, #5 |
| Optimizer state memory/GPU | FP32 (16 B/param) | −30–40% | #4 |
| Total peak memory/GPU | ~budget-limited (mbs=4) | within budget w/ #2+#4 | #2+#4 fund #1+#3 |

*Notes:* Expert memory/GPU rises ~4× (EP=8→2, DP=4) due to replication; the activation + optimizer savings from #2/#4/#5 must cover it — if not, use the TP=2/EP=4 alternative or ZeRO-3 offload. The ~1.9–2.2× target assumes the run is comm/memory-bound (consistent with 62% util); a compute-bound run gains less. "After" numbers are first-cut estimates to validate on the benchmark harness with a small config sweep.

## Risks

- **Convergence (risk of #4):** BF16 optimizer moments can destabilize Adam → loss spikes. Mitigate: keep FP32 master weights, use stochastic rounding, A/B validate loss/perplexity vs. the FP32 baseline at equal step counts.
- **OOM regression (risk of #1, #3):** 4× expert replication plus larger microbatch can blow VRAM. Mitigate: enable checkpointing + seq-parallel first, raise batch incrementally, fall back to EP=4/DP=2 or ZeRO-2.
- **Optimization dynamics shift (risk of #3):** larger microbatch + different grad-accum pattern changes the effective update schedule. Mitigate: hold global batch fixed, monitor loss curves, adjust LR warmup if needed.
- **Recompute overhead (risk of #2):** +10–20% FLOPs hurts if the run is actually compute-bound. Mitigate: use selective recompute and A/B test on/off at the larger batch.
- **Overlap correctness (risk of #6):** async comm introduces stream-race and non-determinism. Mitigate: use validated framework versions, pin comm streams, run determinism checks.
- **Hardware dependence (risk of #7):** grouped-GEMM/NCCL gains vary by GPU and interconnect. Mitigate: gate every change behind the harness sweep so wins are measured, not assumed.

**Suggested validation order:** #4+#2 first (memory headroom) → #3 (batch) → #1 (parallelism) → #5+#6, measuring tokens/sec and util after each step.