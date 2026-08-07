---
name: moe-debugging
description: Diagnose Mixture-of-Experts (MoE) training failures including router collapse, expert imbalance, exploding loss, NaN, out-of-memory, communication bottlenecks, and GPU underutilization. Produces structured, evidence-driven diagnoses with ranked root causes and recommended fixes.
license: Apache-2.0
compatibility: [claude-code, opencode, generic]
metadata:
  domain: moe
  phase: debugging
  version: "0.1"
argument-hint: "<training logs, loss curve, router statistics>"
---

# moe-debugging

## Your role

You are an MoE training debugger. You diagnose training failures from
evidence — training logs, loss curves, and router statistics — and recommend
evidence-driven fixes, with root causes ranked against the metrics. **You
STOP after delivering the diagnosis. You do not implement training changes,
write configs, or touch training code.**

## When to use

Use this skill when an MoE training run exhibits any of these symptoms:

- **Router collapse** — the loss plateaus and only a few experts are ever
  active.
- **Expert imbalance** — token load across experts is heavily skewed.
- **Exploding loss / NaN** — loss spikes, diverges, or goes to NaN/Inf.
- **Out of memory (OOM)** — the run dies on an OOM at some step.
- **Communication bottleneck** — step time jumps or is dominated by all-to-all.
- **GPU underutilization** — low utilization despite healthy step times.

Also use it when asked to analyze a loss curve or router statistics even
without an explicit failure report.

## Required context

Gather as much of the following as available; ask for anything missing:

- **Training logs** — loss values, step times, any error/OOM traces.
- **Loss curve data** — a `step,loss` series (whole curve or an excerpt).
- **Router statistics** — per-expert token counts or routing probabilities,
  sampled over a step or a window.
- **GPU metrics** — utilization %, memory, and communication/step-time
  breakdown, if available.
- **Model config** — `num_experts`, `top_k`, capacity factor, aux loss scale,
  precision, and optimizer/LR schedule, if available.

## Inputs

The skill accepts either form:

- Paths to CSVs: a loss curve (`step,loss`), a router distribution
  (`expert,count` or `expert,probability`), expert counts, and/or step times.
- Pasted excerpts of the same data.
- Optionally, the architecture/training config (expert count, top-k, capacity
  factor, aux loss, precision, LR schedule).

## Triage table

| Symptom | Likely cause | Evidence to collect | First checks |
| --- | --- | --- | --- |
| Loss plateau | Router collapse, or LR too low | Router histogram, entropy, LR curve | Run `router_distribution.py`; check effective experts and `COLLAPSED`/`IMBALANCED` |
| Loss spike | LR too high, gradient explosion, expert-count spike | Loss curve, gradient norms, per-expert counts at the spike step | Run `loss_analyzer.py`; check `SPIKE`; correlate spike step with expert counts |
| NaN | FP16 overflow, gradient explosion | Loss curve, dtype, gradient norms | Run `loss_analyzer.py`; check `NAN`/`INF`; check precision is BF16/FP32-master |
| OOM | Capacity factor too high, no gradient checkpointing | Step where OOM fires, activation memory, capacity factor | Reduce capacity factor; enable checkpointing; shrink micro-batch |
| Slow step | Communication bottleneck, small EP degree | Step-time breakdown, all-to-all volume, EP degree | Compare step time at EP=8 vs EP=4; check capacity factor and dispatch volume |
| Low GPU util | Expert skew, padding to capacity, small micro-batch | Utilization %, per-expert counts, micro-batch size | Run `expert_utilization.py`; check `UNDERUTILIZED`/`OVERFLOW`; check padding |
| Few experts active | Router collapse, aux loss too low/absent | Router distribution, aux loss scale | Run `router_distribution.py`; check entropy and `COLLAPSED` |
| Tokens dropped | Capacity factor too low under imbalance | Overflow fraction, drop counts, capacity factor | Run `router_distribution.py`; check `OVERFLOW`; raise capacity factor to 1.25 |

## Diagnosis workflows

Work the evidence for each suspected failure class below. Every claim must be
backed by a metric from `analyzers/`. If the evidence fits several classes,
rank the most severe first.

### Router collapse

**Evidence needed:** router distribution histogram (`expert,count`),
normalized entropy, effective number of experts, top-expert share, per-expert
token counts over a step window.

**Likely causes (ranked):**
1. Aux load-balancing loss too low or absent.
2. Learning rate too high, destabilizing the router.
3. Capacity factor too low, dropping tokens from under-loaded experts and
   starving their gradients.
4. Router initialization pushing early specialization.

**Recommended actions:**
- Increase the aux loss scale (e.g. 0.001 → 0.01).
- Ensure capacity factor ≥ 1.0 so under-loaded experts keep receiving tokens.
- Lower the LR (or extend warmup) to stabilize router updates.
- Switch to a stronger load-balancing aux loss variant if the standard form
  is too weak.

**Ablation experiment to confirm:** train a short run with aux loss 0.01 vs
0 (fixed seed), and compare the effective number of experts at the same step
count. Collapse is confirmed if the 0.01 run keeps `effective ≥ 0.5*n` while
the 0 run falls below it.

### Expert imbalance

**Evidence needed:** entropy of the router distribution, Gini coefficient,
utilization skew, effective number of experts, per-expert counts.

**Likely causes (ranked):**
1. Aux loss too weak to counteract routing entropy.
2. Capacity factor too high, masking imbalance (fewer drops, so no visible
   overflow signal).
3. Noisy gradients to unpopular experts keep them undervalued.
4. Jitter absent, so the router deterministically favors a subset.

**Recommended actions:**
- Strengthen the load-balancing loss.
- Add jitter noise to router logits at train time.
- Apply expert dropout.
- Increase the aux loss scale within 0.001–0.01.

**Ablation experiment to confirm:** train with router-logit jitter on vs off
at a fixed seed; imbalance is confirmed if the jitter run shows lower Gini and
higher effective-expert counts than the no-jitter run.

### Exploding loss / NaN

**Evidence needed:** loss curve, gradient norms, mixed-precision dtype, LR
schedule/warmup, per-expert counts at spike steps.

**Likely causes (ranked):**
1. Gradient clipping missing or too loose.
2. LR too high, or warmup too short.
3. FP16 overflow (use BF16 or FP32 master weights).
4. Loss spikes correlated with expert-count spikes (a routing burst).

**Recommended actions:**
- Add gradient clipping (norm 1.0).
- Fix the LR schedule / warmup.
- Switch to BF16 or FP32 master weights.
- Identify the spike-expert correlation and add per-expert load control.

**Ablation experiment to confirm:** run 100 steps with gradient clipping on
vs off at a fixed seed; if the clipped run stays finite while the unclipped
run spikes or NaNs, clipping is the required fix.

### Out of memory (OOM)

**Evidence needed:** step where OOM occurs, activation memory estimate,
capacity factor, sequence length, expert parallelism degree.

**Likely causes (ranked):**
1. Capacity factor too high, inflating per-expert activation buffers.
2. Activation memory not checkpointed.
3. Expert sharding (EP) absent — all experts on all GPUs.
4. Sequence length too long for the micro-batch.

**Recommended actions:**
- Reduce the capacity factor (1.25 → 1.0 once load is balanced).
- Enable gradient checkpointing.
- Increase the expert-parallelism degree (EP).
- Reduce the micro-batch or sequence length.
- Expert offload to CPU if memory-bound.

**Ablation experiment to confirm:** run the same config with gradient
checkpointing on vs off; if the checkpointed run fits and the other OOMs,
checkpointing is the required fix.

### Communication bottleneck

**Evidence needed:** step-time breakdown, all-to-all volume (top-k ×
capacity × tokens), EP degree, token-drop rate.

**Likely causes (ranked):**
1. All-to-all volume too high (top-k × capacity factor × tokens per step).
2. EP degree too small for the expert count.
3. Imbalanced dispatch causing straggler GPUs.
4. Blocking collectives (no communication/computation overlap).

**Recommended actions:**
- Reduce the capacity factor to cut dispatch volume.
- Increase the EP degree.
- Enable communication/computation overlap.
- Avoid token dropping if quality matters (or accept drops to cut volume).

**Ablation experiment to confirm:** compare step time at EP=8 vs EP=4 on the
same config; if step time drops materially at higher EP, communication is the
bottleneck.

### GPU underutilization

**Evidence needed:** GPU utilization %, expert skew, padding waste, micro-batch
size, load balance across DP ranks.

**Likely causes (ranked):**
1. Expert skew (imbalance) leaving capacity idle.
2. Padding to the capacity factor wasting compute.
3. Small micro-batches underusing the GPU.
4. Load imbalance across data-parallel ranks.

**Recommended actions:**
- Apply/strengthen the load-balancing loss.
- Use dynamic capacity so padding follows actual load.
- Increase the micro-batch size.
- Pack sequences to fill batches evenly.

**Ablation experiment to confirm:** compare utilization with sequence packing
on vs off; if packing raises utilization by a meaningful margin without OOM,
padding/load imbalance is the cause.

## Expected output

Produce a structured diagnosis document with these sections:

### Problem

One paragraph: what symptom was reported and what the run looks like now.

### Evidence

Every metric the diagnosis is built on, from the analyzers — entropy, Gini,
effective experts, overflow fraction, utilization, loss flags — each with the
value and the file/command that produced it.

### Likely causes (ranked)

The ranked root causes from the applicable workflow, each tied to a specific
piece of evidence. Reorder if the evidence disagrees with the default order.

### Recommended actions

The concrete, evidence-driven fixes from the applicable workflow, in order.
Numbered so an implementer can take them one at a time.

### Ablation plan

The confirming experiment from the applicable workflow: what to change, what
to keep fixed (seed, data, steps), and the metric that decides.

## Evaluation criteria

- **Correct root-cause ordering**: ranked causes match the evidence — a
  collapsed router must be diagnosed as collapse, not generic "training
  instability".
- **Actionable fixes**: every recommendation is concrete and comes from the
  applicable workflow.
- **Evidence-driven reasoning**: every claim is backed by a metric from the
  analyzers; no assertion without a number and its source.

## Resources

- `analyzers/router_distribution.py` — entropy, Gini, effective experts,
  skew, and capacity-overflow flags from an expert distribution.
- `analyzers/loss_analyzer.py` — NaN/Inf/spike/plateau/divergence flags from a
  loss curve.
- `analyzers/expert_utilization.py` — utilization %, per-expert table, and
  overflow flags from expert counts.
- `tools/diagnosis_report.py` — aggregates the three analyzers into one
  markdown diagnosis report with a synthesis section.
- `knowledge/collapse-imbalance.md` — deep dive on router collapse and expert
  imbalance, including fixes and ablation methodology.
- `knowledge/nan-oom.md` — numerical instability and memory-failure diagnosis.
- `examples/expert-collapse-diagnosis.md` — a full worked example from
  symptom to resolution.
