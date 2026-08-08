I'll produce the report. Let me first quickly verify current best practices for MoE throughput tuning to ground the recommendations.
# MoE Training Throughput Optimization Report

## 1. Changes

The baseline (12,400 tok/s, 62% GPU util on 8x GPUs) is limited by three things, in order of impact: **communication-bound expert parallelism**, **memory-bound batch size from FP32 optimizer states**, and **no comm/compute overlap**.

| # | Change | Config (Megatron-style) | Why |
|---|--------|-------------------------|-----|
| 1 | **Reduce `expert_parallel=8 → 2`; add data parallelism `DP=4`** | `--expert-model-parallel-size 2 --data-parallel-size 4` | With EP=8 across only 8 GPUs there is **zero data parallelism**, so every token is all-to-all dispatched to every other rank at every MoE layer. All-to-all can consume 30–60% of total step time (Megatron docs; DeepSeek-V3 shows up to 60%). EP=2 confines dispatch to a 2-rank pair and gives DP=4 for independent batches + ZeRO-style optimizer-state sharding. Rule of thumb: keep TP/EP/PP as small as possible while avoiding OOM. (Fallback: EP=4/DP=2.) |
| 2 | **BF16 optimizer states** (precision-aware optimizer) | `--use-precision-aware-optimizer --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16` | Adam stores 2 moments = 8 B/param in FP32 vs 4 B/param in BF16 — **50% optimizer-state reduction** (~10–12 GB on a ~30B-param model, per Megatron measurements). Computation still runs in FP32 inside the fused Adam kernel, so convergence is essentially unaffected. Frees VRAM that the next change converts into throughput. |
| 3 | **Increase `micro_batch_size=4 → 8`** | `--micro-batch-size 8` (keep `--global-batch-size` constant by halving accumulation steps) | 62% utilization indicates the GPU is memory/compute-underfed. The memory freed in (2) lets us double the micro-batch with roughly the same activation footprint, raising arithmetic intensity (bigger GEMMs, fewer kernel launches) and utilization to ~85%+. |
| 4 | **Overlap EP all-to-all with compute** | `--overlap-moe-expert-parallel-comm` (+ `--delay-wgrad-compute` only if profiling shows it helps) + `CUDA_DEVICE_MAX_CONNECTIONS=2` | With EP=2 and a deeper micro-batch pipeline there is real compute to hide the A2A behind. Note this is workload-sensitive (flat/neutral for small-EP runs), so A/B it before keeping. |
| 5 | **Grouped GEMM for experts** + enable `moe_grouped_gemm`, and tune capacity factor/load-balance loss | `--moe-grouped-gemm`; keep load-balancing loss enabled | Fewer, larger expert GEMMs per layer; prevents a straggler expert (from load imbalance) from stalling the all-to-all rendezvous. |
| 6 | **Selective activation recomputation** (only if needed) | `--recompute-granularity selective` | Not the primary lever here — after changes (2)+(3) memory should fit. Enable *selectively* (layernorm/MoE proj recompute only) only if we want to push the batch to 16. |

Not recommended: activation checkpointing globally (it *reduces* throughput per step), TP>1 (fragments small expert GEMMs), or FP8 on this hardware class (only if Hopper/B200 and needing another 2x).

## 2. Before / After Metrics

Assumes an ~30B-param MoE (≈3B active) on 8× 80GB H100-class GPUs — scale the memory column to the actual model. After = changes 1–4 applied.

| Metric | Before | After | Δ |
|---|---|---|---|
| Tokens/sec | 12,400 | ~28,000 | **+126%** (est. range 24k–32k) |
| GPU utilization | 62% | ~87% | +25 pp |
| Memory / GPU | ~78 GB (FP32 states, no ckpt) | ~68 GB | −10 GB freed by BF16 states |
| Opt. state per param | 8 B (FP32, 2 moments) | 4 B (BF16) | −50% |
| EP dispatch scope | 8 ranks/layer | 2 ranks/layer | −75% A2A volume & latency |

Reasoning for the throughput estimate: the headroom is the gap between 62% and ~90% peak (theoretical 100% util ≈ 20k tok/s at current batch). Doubling the micro-batch + cutting A2A volume 4x + overlapping dispatch should move util to ~85–90% while also raising per-step work, giving the ~2.2x combined gain. Validate with a 100-iteration profile before committing.

## 3. Risks

- **Precision-aware optimizer (BF16 moments):** can slightly change optimization dynamics (moment rounding to BF16). Standard fix: keep master weights and gradient accumulation in FP32/BF16 as configured; run an eval-parity checkpoint (loss/perplexity at same step count) before scaling out. Negligible risk per Megatron/DeepSeek findings when the FP32 cast stays inside the fused kernel.
- **EP reduction changes gradient semantics:** expert gradients must be scaled by `edp_size/dp_size` (different effective batch per expert vs dense layers). With a ChainedOptimizer this is handled, but verify the scaling factor after reconfiguring EP/DP or convergence drifts.
- **Larger micro-batch / accumulation change:** global batch is kept constant, but the LR schedule (warmup steps, cosine) may need a small retune; watch for optimizer state drift if accumulation order changes (FP ordering).
- **EP A2A overlap is workload-sensitive:** for small EP with `alltoall` on ≤2 nodes it can be flat or slightly *slower* (scheduling overhead exceeds the hidden communication). It also asserts with full recompute, PP without VPP, and shared-expert overlap — keep these mutually exclusive and A/B test.
- **Communication movement:** reducing EP lowers inter-GPU traffic, but DP=4 adds gradient reduce-scatter. If the node is PCIe-only (no NVLink), DP communication can become the new bottleneck; enable `--overlap-grad-reduce` / `--overlap-param-gather` with the distributed optimizer.
- **Load imbalance / capacity factor:** if expert utilization is skewed, per-GPU compute idles inside the A2A barrier. Tune capacity factor + load-balancing loss; do not drop tokens aggressively (hurts quality).
- **Validation risk:** all `After` numbers are estimates; 62%→87% util assumes the GPU was memory/compute-underfed rather than host/launch-bound. Profile (Nsight) first — if kernel-launch-bound, prioritize CUDA graphs + fewer small kernels over batch size.