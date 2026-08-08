The web search results gave me solid grounding (Megatron-Core MoE optimization docs, MegaScale-MoE paper) for the diagnosis and expected gains. Here is the report.

---

# MoE Training Throughput Report — 8×GPU Run

**Baseline:** 12,400 tokens/sec · 62% GPU utilization · `expert_parallel=8` (DP=1) · `micro_batch_size=4` · no activation checkpointing · FP32 optimizer states

## Changes

**Root-cause diagnosis:** At EP=8 with no DP, every MoE layer performs a full 8-GPU all-to-all dispatch — this is the dominant wall (Megatron-Core reports EP all-to-all can consume 30–40% of step time). Expert GEMMs are also small (few tokens per expert) so SMs are under-fed, and FP32 optimizer states + full activation retention force a small micro-batch. Concretely, per GPU: model params (BF16) 2B/param + FP32 master 4B + Adam m/v 8B = 14B/param, versus 6B/param with BF16 optimizer states.

| # | Change | Rationale |
|---|--------|-----------|
| 1 | **Optimizer states FP32 → BF16** (precision-aware optimizer, `--exp-avg-dtype bf16 --exp-avg-sq-dtype bf16`) | Cuts weight+optimizer memory ~57% (14→6 B/param). No throughput change by itself — it buys the memory headroom for changes 3–4. |
| 2 | **Enable selective activation recompute** (`--recompute-granularity selective --recompute-modules layernorm moe_act up_proj`) — *not* full-layer | Full checkpointing in MoE re-triggers the EP all-to-all on every recompute (~33%+ overhead). Selective recompute of cheap ops saves most activation memory at only ~2–5% overhead. Frees memory, enables larger micro-batch. |
| 3 | **Increase `micro_batch_size` 4 → 8** | More tokens per expert per step → larger expert GEMMs → higher SM utilization, and better amortization of the all-to-all and kernel launches. Enabled by memory freed in changes 1–2. |
| 4 | **Overlap EP communication** (`--overlap-moe-expert-parallel-comm --delay-wgrad-compute`) | Overlaps all-to-all dispatch with expert computation (and fwd/bwd across microbatches). Directly attacks the 30–40% comm wall — the single biggest expected win. |
| 5 | **Enable kernel fusions** (`--moe-grouped-gemm --moe-router-fusion --moe-permute-fusion`) | Grouped GEMM batches skinny per-expert GEMMs into one kernel; router/permute fusions cut launch count and latency — directly raises GPU utilization. |
| 6 | **Reduce host/kernel-launch overhead** (`--manual-gc --manual-gc-interval 100`; CUDA graphs on router/preprocess) | Fills inter-kernel gaps; manual GC stops Python GC pauses that produce utilization dips. |
| 7 | *(Optional, further gain)* **EP 8 → 4 with DP=2** + `--use-distributed-optimizer` | Halves all-to-all volume, doubles per-GPU expert GEMM size, and DP=2 lets grad-reduce overlap. Cost: attention params replicated ×2. Do only if #4 doesn't close the gap. |
| 8 | *(Optional, highest gain/risk)* **FP8 expert GEMMs** (`--fp8-format e4m3 --fp8-recipe blockwise`) | Faster tensor-core GEMMs and smaller activation buffers. Quantization risk (see below). |

## Before/After Metrics

Projected values (no instrumented run yet — based on published Megatron-Core / MegaScale-MoE gains of ~1.3–1.9× from comm overlap + fusions).

| Metric | Before | After (projected) | Delta |
|--------|--------|-------------------|-------|
| Throughput (tokens/sec) | 12,400 | ~17,000–19,500 | **+37–57%** |
| GPU utilization | 62% | ~78–85% | **+16–23 pp** |
| Peak memory / GPU | ~100% (limit) | ~80–90% | **−10–20% headroom** (reinvested in MBS) |
| Micro-batch size | 4 | 8 | 2× |
| Optimizer state bytes/param | 14 | 6 | −57% |
| Comm (all-to-all) time share | ~30–40% of step | ~5–15% (overlapped) | large reduction |

## Risks

- **Selective recompute still adds compute** (~2–5%); if GEMMs are already the bottleneck it yields net memory savings but flat throughput — measure with profiler before/after.
- **BF16 optimizer states can degrade convergence** for outlier-heavy gradients; monitor loss/accuracy curves and fall back to FP32 for specific layers if needed.
- **FP8 (optional #8)** risks precision loss, requires loss-scaling care, and can shift the wall to quantization kernels on smaller matrices; validate loss parity over a few hundred steps before adoption.
- **MBS 4→8 raises peak activation + expert-buffer memory and worsens expert load imbalance** (all-to-all imbalance grows with tokens per batch). Watch expert capacity/utilization; add auxiliary-load-balancing loss or cap tokens per expert if idle-expert skew appears.
- **Communication overlap (#4) adds scheduling complexity and buffer pressure**; on some configurations it can conflict with dropless routing or CUDA graphs (static shapes) — test combinations, not all flags at once.
- **EP 8→4 (#7) replicates attention weights** (memory ↑) and changes the checkpoint layout; requires `--use-distributed-optimizer` to offset, and is a larger architectural change than the others.
- **Numbers above are projections.** Profile after each change (Nsight trace / step-time breakdown) before stacking more; the three walls (memory / comm / compute) interact — fixing one can expose the next.

**Suggested roll-out order:** 1 → 2 → 3 → 5/6 → 4 → (8) → (7), re-profiling after each.