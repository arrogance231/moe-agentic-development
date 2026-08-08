This maps to the `moe-performance` skill. Here is the report.

---

# MoE Throughput Optimization Report

## Baseline Context

Existing run: **12,400 tokens/sec** on 8 GPUs, **62% GPU utilization**, `expert_parallel=8`, `micro_batch_size=4`, **no activation checkpointing**, **FP32 optimizer states**.

The 62% utilization with a micro-batch of 4 indicates the GPUs are compute-starved: matmul shapes are too small to saturate tensor cores, and activation memory is being spent on un-checkpointed buffers rather than batch size. The ranking below is ordered by ROI.

## Changes

### 1. Enable activation checkpointing (enabler)
- **Change:** Set `checkpoint_activations=true` (recompute in backward).
- **Why:** Halves activation memory (~`num_layers × micro_batch × seq_len × d_model × 20 / tp`), freeing the headroom needed for Change 2 without OOM. Activations are the dominant memory consumer at `seq_len > 2048`, and there is none today.
- **Expected gain:** Enables the micro-batch increase below; on its own costs +20–30% FLOPs from recompute.

### 2. Raise `micro_batch_size` from 4 → 8
- **Change:** `micro_batch_size=8`, keep `grad_accum` unchanged so the global batch stays fixed (`global_batch = micro_batch × grad_accum × dp`).
- **Why:** 4 is far below the **8–64 tokens-per-expert-per-GPU floor** (`micro_batch × seq_len × top_k / (num_experts / ep)` with `ep=8`). Doubling the micro-batch doubles expert FFN matmul size, which is the largest compute block in an MoE layer and the most sensitive to tensor-core efficiency. 62% → ~85%+ util is the direct payoff.
- **Memory check:** checkpointed (halved) activations × doubled micro-batch ≈ current activation budget — net flat. If OOM persists, raise `ep` or shave 10% off capacity factor.
- **Expected gain:** **+35–40% tokens/sec** (12,400 → ~17,000), from closing the utilization gap.

### 3. Mixed-precision matmuls (BF16 forward/backward, FP32 master + optimizer states)
- **Change:** Run matmuls in BF16 (tensor cores) while keeping FP32 master weights and FP32 optimizer states — do **not** drop to FP8 yet.
- **Why:** If matmuls currently execute in FP32, they bypass tensor cores; BF16 gives +20–40% on compute-bound regions. This is the second-largest lever after batch size. It also trims weight/gradient working-set bytes (2B vs 4B), adding memory slack for Change 2.
- **Risk note:** FP32 optimizer states are retained for numerical safety; FP8 is deferred (see Risks).

### 4. (Conditional) Sequence packing / capacity-factor tune — **only if router stats show skew or padding**
- **Change:** Pack sequences to fill capacity exactly; once load is balanced, drop capacity factor toward 1.0.
- **Why:** Padding to capacity factor wastes compute and inflates all-to-all volume. Only apply if `router_distribution.py` shows imbalance/overflow flags — no router data was supplied, so this is listed as conditional, not unconditional.
- **Expected gain:** +10–15% tokens/sec if padding waste is confirmed.

## Before / After Metrics

| Metric | Before | After (projected) | Delta |
| --- | --- | --- | --- |
| Tokens/sec (global) | 12,400 | **~17,000** | **+37%** |
| GPU utilization (%) | 62% | **85–90%** | **+23–28 pp** |
| Activation memory/GPU | baseline (un-checkpointed) | ~halved by checkpointing, ~neutralized by doubled micro-batch | flat–lower |
| Peak memory/GPU | FP32 weights+grads full-size | BF16 weights/grads (2B) + FP32 master/moments | lower (BF16 halves weight/grad working set) |
| Expert FFN matmul size | small (mbs=4) | 2× (mbs=8) | — |

Projections assume gains land in the skill's documented bands: micro-batch/larger-matmuls (+10–30% util), BF16 tensor-core (+20–40% on compute-bound regions), checkpointing (+20–30% compute but net-positive from higher saturation). Actual numbers must be confirmed with `profilers/throughput_profiler.py` before/after; the delta is a trend, not a guarantee.

## Risks

- **Micro-batch increase → OOM.** Doubling mbs doubles activation buffers; mitigation is Change 1 (checkpointing) applied first, verified with `tools/memory_estimator.py` (target ≥20% headroom) before the batch change lands.
- **Gradient checkpointing adds +20–30% compute.** If the kernel/overlap overhead exceeds the utilization gain at small batch sizes, throughput could regress — A/B: run `mbs=8` with checkpointing vs `mbs=4` without, same seed/steps, compare tokens/sec.
- **BF16 precision loss.** FP32→BF16 can destabilize the router (updates noisier). Mitigation: keep FP32 master weights and FP32 optimizer states, monitor loss/NaN via `loss_analyzer.py` for the first 100 steps. **Do not** move to FP8 without scaling calibration — that is future scope.
- **All-to-all growth with bigger batches.** `ep=8` on 8 GPUs means one all-to-all per MoE layer; doubling the micro-batch doubles dispatch volume. Low risk now (comm share is small at 62% util) but re-check step-time breakdown; if comm share climbs, trade to `ep=4` or trim capacity factor.
- **Conditional optimizations unverified.** Changes 4 lacks router data; applying it without evidence risks distorted routing. Confirm imbalance with `router_distribution.py` first.

## Verification plan

1. Run `profilers/throughput_profiler.py` on the current config to baseline step-time stats, tokens/sec, GPU-util proxy, and MFU.
2. Apply Change 1 → 2 → 3 one at a time; re-run the profiler after each so gains are attributable.
3. Confirm memory with `tools/memory_estimator.py` before and after each step (target ≥20% headroom).
4. Run 100 steps at a fixed seed before/after to confirm loss trajectory is stable (no NaN/spike via `loss_analyzer.py`).