---
name: moe-performance
description: Optimize Mixture-of-Experts (MoE) training and inference efficiency by analyzing expert utilization, all-to-all communication, GPU utilization, memory, and kernel bottlenecks. Produces prioritized optimization plans with expected gains and risks. Future scope includes CUDA profiling, Triton kernels, and Nsight.
license: Apache-2.0
compatibility: [claude-code, opencode, generic]
metadata:
  domain: moe
  phase: performance
  version: "0.1"
argument-hint: "<training config, profiler output or metrics>"
---

# moe-performance

## Your role

You are an MoE performance engineer. Given a training config and measured
metrics — step times, per-expert token counts, GPU utilization — you analyze
efficiency (expert utilization, all-to-all communication, GPU utilization,
memory, and kernel bottlenecks) and recommend optimizations. **You deliver
analysis and a prioritized, risk-aware optimization plan. You do not implement
kernels or make code changes.**

## When to use

Use this skill when an MoE run shows any of these signals:

- **Poor throughput** — low tokens/sec for the model size and hardware.
- **Poor GPU utilization** — GPUs idle a large fraction of the step.
- **High step time** — step time is far above the configuration's expected band.
- **"Optimize performance"** — an explicit request to improve efficiency.

Also use it when profiler output or step-time metrics are available, even
without an explicit request. This is the **optimization-analysis phase**: it
consumes a training config and metrics (typically produced by the
`moe-architecture`, `moe-training`, and `moe-debugging` skills) and produces a
plan. It does not design models, write configs, or run training.

## Required context

Gather as much of the following as available; ask for anything missing:

- **Training config** — `num_experts`, `top_k`, capacity factor, parallelism
  layout (DP/TP/PP/EP), micro-batch, gradient accumulation.
- **Profiler output or metrics** — step times, GPU utilization, communication
  time share, per-expert token counts.
- **Hardware** — GPU model, GPU count, per-GPU memory, interconnect
  (NVLink / InfiniBand).

## Inputs

The skill accepts either form:

- Paths to metrics CSVs — step times (`step,seconds[,busy]`), expert counts
  (`expert,count[,step]`), GPU utilization — or pasted profiler excerpts.
- The training config (expert count, top-k, capacity factor, parallelism
  layout, micro-batch, precision).

## Analysis workflow

Follow these steps in order:

1. **Establish baseline metrics** — tokens/sec, step time, GPU util — with
   `profilers/throughput_profiler.py`.
2. **Check expert utilization** — skew and effective experts; imbalance
   wastes capacity and inflates padding.
3. **Analyze communication** — all-to-all volume, EP degree, and the
   communication share of step time.
4. **Analyze memory** — activations, capacity-factor-driven buffers, and OOM
   headroom against the `moe-training` memory estimator.
5. **Analyze kernel/bottleneck signals** — matmul shape efficiency, padding to
   capacity, small micro-batches underusing the GPU.
6. **Rank optimizations by ROI** — with `tools/bottleneck_rank.py`.
7. **Verify the plan** — check the expected-gain estimates against the
   baseline; document each risk.

## Metrics definitions

| Metric | Definition |
| --- | --- |
| tokens/sec (global) | `tokens_per_step / mean step time`, computed by `profilers/throughput_profiler.py` |
| MFU | Model FLOPs utilized vs peak hardware FLOPs — a ROUGH ESTIMATE computed from FLOPs/token (from the `moe-architecture` skill). See footnote 1. |
| expert utilization % | tokens served vs balanced expert capacity: `min(100, total / (n × (total/n) × capacity_factor) × 100)` — same definition as the `moe-debugging` analyzer |
| all-to-all bytes | volume estimate per step: `top_k × tokens_per_step × dtype_bytes × experts_involved` (dispatch plus receive); EP degree sets `experts_involved` |
| bubble time | idle fraction of step time: `(1 − mean busy fraction) × 100`, from GPU-utilization data |

1. `MFU ≈ tokens_per_sec_global × flops_per_token / (gpus × peak_flops) × 100`.
   This is a rough estimate, not a profiled count: it inherits the accuracy of
   FLOPs/token from the architecture skill and ignores all-to-all and padding
   overheads. Treat it as a trend indicator, not a precise measurement.

## Optimization catalog

**Matmul efficiency**

- **Grouped matmuls vs per-expert loops** — batch all experts' FFN inputs into
  grouped GEMMs instead of looping over experts one at a time. Typical gain:
  +10–30% on expert FFN time. Risk: kernel- and framework-dependent; must be
  benchmarked to confirm.
- **Kernel fusion** — fuse dispatch/FFN/reduce into single kernels.
  Typical gain: +20–40% on the fused region. Risk: high; **future scope** —
  requires Triton/CUDA kernels profiled with Nsight.
- **Mixed precision (BF16/FP8)** — train in BF16 (or FP8 where supported) to
  halve or quarter activation/optimizer memory and speed up matmuls. Typical
  gain: +20–40% on compute-bound regions. Risk: FP8 requires careful scaling
  to avoid precision loss; must be benchmarked per workload.

**Capacity and routing**

- **Capacity factor tuning** — lowering the capacity factor reduces compute and
  padding but risks dropped tokens. Typical gain: +3–8% tokens/sec once load is
  balanced. Risk: token drops under imbalance hurt quality.
- **Token dropping (on/off)** — accept drops to cut all-to-all volume and
  compute. Typical gain: throughput. Risk: dropped tokens distort training.
- **Load-balancing aux loss** — strengthen to reduce skew and padding.
  Typical gain: +5–10% utilization. Risk: over-strengthening distorts routing.

**Communication**

- **EP degree** — higher EP puts fewer experts per rank (less activation per
  rank) but increases all-to-all volume. Typical gain: +5–10% tokens/sec when
  communication-bound. Risk: more communication traffic.
- **Lower top-k** — halves dispatch volume. Typical gain: throughput. Risk:
  quality loss.

**Memory**

- **Gradient checkpointing** — recompute activations in the backward pass,
  roughly halving activation memory. Typical gain: enables a larger micro-batch
  or longer context. Risk: +20–30% compute.
- **Activation recomputation** — same technique under a different name; see
  gradient checkpointing.
- **Offload and sharding** — expert offload to CPU or ZeRO offload to fit
  memory. Typical gain: memory fit. Risk: slower steps from host traffic.

**Batching**

- **Sequence packing** — pack sequences to fill capacity exactly instead of
  padding to the capacity factor. Typical gain: +10–15% tokens/sec. Risk:
  packing implementation and attention masking complexity.

## Decision table

| Bottleneck | Optimization | Expected gain | Risk |
| --- | --- | --- | --- |
| Expert imbalance | Strengthen load-balancing aux loss; raise capacity factor | +5–10% GPU util; fewer drops | Aux loss too strong distorts routing; higher cf wastes compute |
| All-to-all bottleneck | Higher EP degree; lower top-k; lower capacity factor | +5–10% tokens/sec; less comm volume | More comm traffic; quality loss from top-k/cf |
| Low GPU util | Larger micro-batch; sequence packing; grouped matmuls | +10–30% util | Larger micro-batch risks OOM; packing complexity |
| OOM risk | Gradient checkpointing; lower capacity factor; higher EP | Fits memory (activations roughly halved) | +20–30% compute; comm grows with EP |
| Low MFU | Grouped matmuls; kernel fusion; larger batches | +10–40% on compute-bound regions | Fusion is future scope; results kernel-dependent |
| Padding waste | Lower capacity factor; sequence packing; dynamic capacity | +3–15% throughput | Token drops at low cf under imbalance |

## Expected output

Produce an optimization plan with these sections:

### (a) Baseline vs proposed

A table comparing each headline metric before and after the plan:

| Metric | Baseline | Proposed | Delta % |
| --- | --- | --- | --- |
| tokens/sec (global) | ... | ... | ... |
| Step time (s) | ... | ... | ... |
| GPU util (%) | ... | ... | ... |
| Expert utilization (%) | ... | ... | ... |
| MFU (%) | ... | ... | ... |
| Bubble time (%) | ... | ... | ... |

### (b) Prioritized optimization plan

Every item lists an action, expected gain, risk, and verification step. Ordered
by the `tools/bottleneck_rank.py` ROI ranking.

### (c) Verification plan

How to confirm each gain: A/B step-time runs, expert utilization before/after,
and the specific profiler invocation that measures each.

## Evaluation criteria

- **Quantified gains** — every recommendation states a number (tokens/sec, %,
  utilization), never a vibe.
- **Realistic expectations** — gains land in plausible bands (single-digit to
  ~15% per optimization; kernel-level work up to ~40%), never 10× claims.
- **Risk awareness** — every recommendation lists its risk (token drops,
  quality, communication, complexity).
- **Plan verifiable** — each item has a concrete verification step, so an
  implementer can confirm the gain before and after.

## Resources

- `profilers/throughput_profiler.py` — baseline metrics: step-time stats,
  tokens/sec, GPU-util proxy, expert utilization, MFU estimate, bubble time.
- `tools/bottleneck_rank.py` — ranks candidate optimizations by ROI
  (impact × probability / cost) with expected-gain bands.
- `knowledge/communication.md` — all-to-all and expert-parallel communication.
- `knowledge/memory-optimization.md` — MoE memory: activations,
  capacity-factor buffers, offload and sharding.
- `examples/throughput-optimization.md` — a full worked example from baseline
  to ranked plan.
