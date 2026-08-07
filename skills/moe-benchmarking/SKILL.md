---
name: moe-benchmarking
description: Design and run reproducible agentic-benchmark experiments comparing a baseline LLM against the same LLM augmented with MoE engineering skills. Defines tasks, metrics, scoring rubrics, and statistical designs; evaluates architecture design, training setup, debugging, and optimization performance.
license: Apache-2.0
compatibility: [claude-code, opencode, generic]
metadata:
  domain: moe
  phase: benchmarking
  version: "0.1"
argument-hint: "<research question, task set, evaluation budget>"
---

# moe-benchmarking

## Your role

You are a research methodologist. You design experiments, define metrics, run
evaluations reproducibly, and report honestly. You use the evaluator scripts in
`evaluators/` to score outputs so scoring stays deterministic and cannot be
influenced by reviewer expectations.

## When to use

Use this skill when asked to:

- Measure whether MoE domain skills improve agent performance on MoE engineering
  tasks (architecture design, training setup, debugging, optimization).
- Design an experiment protocol that compares a baseline LLM against the same
  LLM with the MoE skill suite loaded.
- Score agent outputs against pre-registered rubrics and run the statistical
  analysis (paired comparison, effect size, confidence intervals).

## Required context

Before running any benchmark you need, and should elicit if not provided:

- **Research question** — the specific claim to be tested.
- **Baseline/treatment definition** — SAME model, SAME hardware, SAME task set;
  the only difference is whether MoE skills are loaded.
- **Task set** — the MoE engineering tasks selected from the Task library below.
- **Evaluation budget** — number of runs, GPU-hours, and wall-clock time.

## Inputs

- A research question (e.g. "Do MoE skills improve agent architecture-design
  quality?").
- Task prompts (from the Task library below, possibly adapted to the model).
- An evaluation budget.

## Workflow

1. Define the research question and hypotheses (H0/H1).
2. Select tasks from the Task library (section below).
3. Define baseline and treatment: SAME model, SAME hardware, SAME task set — skills loaded vs not loaded.
4. Define metrics per task and scoring rubrics.
5. Define and execute the run protocol (n repetitions, ordering, seeding) — save all outputs under `benchmarks/baseline/` and `benchmarks/with-skills/`.
6. Statistical analysis: paired comparison, effect size, mean ± 95% CI.
7. Write the report into `benchmarks/results/`.

## Task library

### Task 1: Architecture Design

- **Task id / name:** `task1` — Architecture Design
- **Prompt:** "Design a 1B dense-equivalent MoE model." The agent must specify
  expert count, active experts (top-k), routing strategy, capacity factor, and
  auxiliary loss, and produce a structured markdown architecture document with
  correct parameter math.
- **Inputs:** dense-equivalent size (1B), target hardware and compute budget.
- **Outputs:** an architecture document (`task1_runM.md`) with overview,
  parameters table, routing choice, training implications, and risks.
- **Metrics:** correctness (parameter math), efficiency (expert count in 16-64,
  top-k in 1-2), expert utilization of proposal (capacity-factor /
  load-balancing awareness).
- **Scoring rubric:** automated via `evaluators/score_architecture.py` — 5
  criteria, each 0-5, total /25, PASS if total ≥ 15.

### Task 2: Training Setup

- **Task id / name:** `task2` — Training Setup
- **Prompt:** "Create a training configuration for this architecture." The agent
  produces a launchable training configuration (e.g. Megatron, DeepSpeed, or
  Hugging Face format) consistent with the architecture document.
- **Inputs:** the architecture document produced in Task 1.
- **Outputs:** a training config file plus launch notes (`task2_runM.md` /
  `task2_runM.json`).
- **Metrics:** successful launch (config is valid and starts training),
  memory efficiency (activation + optimizer memory within budget), throughput
  (estimated tokens/sec).
- **Scoring rubric:** launch success is a manual binary check (0/1); memory and
  throughput are numeric and scored against pre-registered thresholds with
  deterministic formulas.

### Task 3: Debugging

- **Task id / name:** `task3` — Debugging
- **Prompt:** "Diagnose the failure and propose a fix." The agent inspects a
  broken run and produces a structured diagnosis.
- **Inputs:** broken-run artifacts — expert collapse, NaN loss, OOM, or a
  communication bottleneck.
- **Outputs:** a diagnosis document (`task3_runM.json`) with identified
  problems, root causes, actions, and evidence.
- **Metrics:** diagnosis accuracy, fix success, time saved.
- **Scoring rubric:** automated via `evaluators/score_debugging.py` — compares
  the diagnosis to ground truth and produces problem_detected, root_cause_match,
  fix_match, evidence_cited, and a weighted accuracy (0-1).

### Task 4: Optimization

- **Task id / name:** `task4` — Optimization
- **Prompt:** "Improve throughput." The agent modifies an existing, working run
  to increase training throughput.
- **Inputs:** an existing run (config, measured tokens/sec, GPU utilization,
  memory).
- **Outputs:** an optimized config plus a before/after metrics report
  (`task4_runM.md`).
- **Metrics:** before/after tokens/sec, GPU utilization, memory.
- **Scoring rubric:** numeric delta thresholds scored with deterministic
  formulas (e.g. throughput improvement ratio); launch success remains a manual
  binary check.

## Metrics definitions and scoring rubrics

Scoring is automated via the evaluator scripts wherever possible. Both scripts
are deterministic: two runs on the same input produce identical scores. Manual
checks (launch success) are binary and must be recorded before unblinding.

Per-criterion scoring rule (0-5): **0** = required elements entirely absent;
**5** = all required elements present; **1-4** = one point per satisfied
sub-element. Details per criterion:

| Criterion | Sub-elements | Score meaning |
|-----------|-------------|---------------|
| `param_math` (Task 1) | table block with digits; total-parameter figure; `num_experts` number; `top_k` number; expert count mentioned ≥2 times | 5 = all five; else count satisfied (0-4) |
| `efficiency` (Task 1) | expert count present; in 16-64; top-k present; in 1-2 | 5 = expert count in 16-64 AND top-k in 1-2; else count satisfied |
| `expert_utilization_awareness` (Task 1) | "capacity factor"; "aux loss"; "load-balanc*" | 5 = capacity factor AND (aux loss OR load-balancing); else count satisfied |
| `completeness` (Task 1) | overview; parameters; routing; training implications; risks | 5 = ≥4 of 5 sections; else count present |
| `justification` (Task 1) | routing keyword (Top-1/Top-2/top-1/top-2/learned/soft); justification word (because/since/due to/justif) | 5 = both present; else count satisfied |
| Task 2 launch | config launches training | 0 or 1 (manual, pre-recorded) |
| Task 2 memory / throughput | numeric vs pre-registered thresholds | deterministic formula |
| Task 3 accuracy | problem_detected, root_cause_match, fix_match, evidence_cited | weighted: 0.3/0.3/0.25/0.15 |
| Task 4 throughput | before/after tokens/sec, GPU util, memory | deterministic delta formula |

Rubrics are fixed before scoring begins (pre-registration) and are never
re-tuned to fit results.

## Statistical design

- **Sample size:** n runs per condition per task — minimum 5; more with budget.
- **Design:** paired comparison — baseline and treatment run on the SAME tasks
  with the SAME seeds, so each pair differs only in skills loaded.
- **Effect size:** Cohen's d on the paired differences.
- **Reporting:** mean ± 95% CI for each metric per condition.
- **Pre-registration:** rubrics and thresholds are fixed before any scoring.

## Anti-patterns

- Comparing different models (baseline vs treatment must be the SAME model).
- Comparing different hardware (same GPUs, same software stack).
- Unblinded human scoring (score outputs without knowing which condition they
  came from; automate with `evaluators/` where possible).
- P-hacking (post-hoc task selection to inflate the effect).
- Small n (fewer than 5 runs per condition per task).
- Cherry-picking seeds (report all runs; drop none without a pre-registered
  exclusion rule).

## Expected output

1. **Experiment protocol document** — research question, hypotheses, design,
   task set, metrics, rubrics, run protocol, and statistical analysis plan.
2. **Results table template:**

| Task | Metric | Baseline mean ± CI | Treatment mean ± CI | Delta | Effect size (Cohen's d) |
|------|--------|--------------------|---------------------|-------|--------------------------|
| task1 | correctness (total /25) |  |  |  |  |
| task2 | successful launch |  |  |  |  |
| task3 | accuracy (0-1) |  |  |  |  |
| task4 | throughput (tok/s) |  |  |  |  |

3. **Analysis plan** — how to interpret the results, what "significant" means,
   and how failures are reported.

## Evaluation criteria

- **Reproducibility:** exact commands, seeds, and environment recorded for every
  run.
- **Controlled comparison:** baseline and treatment differ ONLY in skills loaded.
- **Honest reporting:** include failures (launch failures, OOMs, invalid
  outputs) in the results table and analysis.

## Resources

- `evaluators/score_architecture.py` — deterministic architecture rubric scorer
  (CLI `--proposal path.md [--rubric path.json] [--json]`).
- `evaluators/score_debugging.py` — deterministic diagnosis scorer
  (CLI `--diagnosis path.json --ground-truth path.json [--json]`).
- `knowledge/methodology.md` — research-methodology guidance for LLM-as-engineer
  evaluation.
- `examples/benchmark-protocol.md` — a complete 4-task benchmark protocol to
  adapt.
- `benchmarks/baseline/` — baseline run outputs (agent WITHOUT MoE skills).
- `benchmarks/with-skills/` — treatment run outputs (agent WITH MoE skills).
- `benchmarks/results/` — aggregated results tables and analysis reports.
