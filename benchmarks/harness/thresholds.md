# Pre-registered numeric thresholds — task2, task4, task6

BENCHMARK.md / methodology.md / benchmark-protocol.md list task2's "memory
efficiency" and "throughput" and task4's "throughput improvement" and "GPU
utilization/memory" as **numeric threshold** / **numeric delta** metric
*types* without giving concrete numeric values anywhere in the repo (grep
confirms: no `%`, `tok/s`, `GB`, or threshold-table entries exist beyond the
column headers themselves). These are placeholders the runner is expected to
pre-register before scoring, per methodology.md's "pre-register the rubrics
... before running" instruction. Fixed here, before any full-wave run, so
scoring cannot be tuned post hoc to any particular arm's output.

## task2 — Training Setup ("successful launch, memory efficiency, throughput")

task2's prompt (see task_prompts.py) does not pin a GPU count or VRAM budget
(agents choose their own parallelism), so an absolute VRAM ceiling can't be
pre-registered without inventing a constraint the prompt never gave. Score
what's mechanically checkable instead:

- **successful launch (0/1)**: output parses as JSON matching the required
  schema (framework, architecture{num_experts, top_k, hidden_size,
  num_layers, capacity_factor}, parallelism{data_parallel, expert_parallel,
  tensor_parallel}, batch_size, estimated_memory_gb,
  estimated_tokens_per_sec, launch_notes) — all keys present, correct types,
  and internally consistent with task1/task2's own stated architecture
  (num_experts * per-expert-params + shared params ≈ 1B, top_k ≤ num_experts).
  Fails (0) on missing/malformed keys or an inconsistent param count.
- **memory efficiency (ratio, informational)**: `estimated_memory_gb /
  (num_gpus_implied_by_parallelism * 80)`, where num_gpus_implied =
  data_parallel * expert_parallel * tensor_parallel and 80 is the standard
  MI300X-class VRAM figure used elsewhere in this run's environment
  provenance. Reported as a ratio per arm, not pass/failed — BENCHMARK.md
  doesn't define a target ratio, so this is descriptive only.
- **throughput (tok/s, informational)**: `estimated_tokens_per_sec` reported
  verbatim per arm. This is a self-reported number from the agent (no actual
  training run happens), so it's an internal-consistency / plausibility
  signal, not ground truth — flagged as a validity limitation in summary.md.

## task4 — Optimization (baseline given IN the prompt: 12,400 tok/s, 8 GPUs, 62% util)

task4's prompt states a concrete baseline (12,400 tok/s, 62% GPU util,
expert_parallel=8, micro_batch_size=4, no activation checkpointing, FP32
optimizer states). Score the *claimed* delta against that baseline:

- **throughput improvement**: `(after_tok_s - 12400) / 12400`, parsed from
  the agent's own Before/After table. A claim is only counted if the
  proposed changes are causally plausible (activation checkpointing tradeoff
  correctly described as memory-for-compute, BF16/FP16 optimizer state
  correctly described as a memory reduction, etc.) — implausible or
  internally contradictory claims (e.g. "reduced memory AND increased batch
  size with same GPU count" with no explanation) are flagged, not
  auto-failed, since this is self-reported and not independently verified.
- **GPU utilization delta**: `after_util_pct - 62`, reported as-is.
- Both are **self-reported deltas, not measured** (no actual training run
  executes) — this is stated explicitly in summary.md as a validity
  limitation, consistent with the "cost-metric non-portability" and
  self-report threats already flagged in methodology.md.

## task6 — Constrained-hardware design

Reuses `score_architecture.py`'s five criteria (param_math, efficiency,
expert_utilization_awareness, completeness, justification, 0-25 total,
PASS ≥15) plus one additional deterministic **constraint-fit check** (0/1),
since the prompt's constraint (25 Gbps interconnect, expert_parallel degree
fixed at 4) is directly checkable from the required output fields:

- **constraint-fit (0/1)**: proposal states expert_parallel = 4 explicitly
  (regex `expert.parallel.{0,20}4` or an explicit "EP=4"/"EP degree: 4"
  statement) AND states a communication-volume figure or bound AND
  addresses the 25 Gbps budget explicitly (mentions "25 Gbps", "Gbps", or an
  equivalent bandwidth unit in the same section as the communication
  discussion). All three required or the check fails — a proposal that
  ignores the interconnect constraint entirely should not pass regardless of
  its architecture-criteria score.
- **task6 total** reported as (score_architecture total /25, constraint_fit
  0/1) — two separate columns in results-task6.md, not summed, since they
  measure different things (architecture quality vs. constraint compliance).

## Decision rationale (for run-log.md)

Made the most defensible call per the plan's "if ambiguous, log reasoning
and continue" instruction: task2 and task4's numeric fields are
self-reported by the agent under test (no real training run happens in this
benchmark), so they're scored as descriptive/consistency signals rather than
independently verified ground truth. This is a real limitation of the
benchmark design as currently documented (BENCHMARK.md never specifies that
an actual training job executes) and is stated plainly in summary.md rather
than papered over.
