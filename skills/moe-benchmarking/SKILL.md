---
name: moe-benchmarking
description: Design and run reproducible agentic-benchmark experiments comparing a search-enabled baseline LLM against the same LLM augmented with MoE engineering skills. Defines the four-arm design (bare, search, skills, skills+search), tasks, quality/cost/reliability metrics, scoring rubrics, and statistical designs; evaluates architecture design, training setup, debugging, and optimization performance.
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
  tasks (architecture design, training setup, debugging, optimization) **over a
  search-enabled baseline** — not over an information-starved one.
- Design an experiment protocol that compares a retrieval-capable baseline LLM
  against the same LLM with the MoE skill suite loaded.
- Score agent outputs against pre-registered rubrics and run the statistical
  analysis (paired comparison, effect size, confidence intervals).

## Required context

Before running any benchmark you need, and should elicit if not provided:

- **Research question** — the specific claim to be tested.
- **Arm definitions** — SAME model, SAME hardware, SAME task set, SAME seeds;
  arms differ only in whether retrieval is registered as a tool and whether MoE
  skills are loaded (see "Experimental arms" below).
- **Retrieval configuration** — which search/doc-retrieval backend the
  search-enabled arms use, and the per-run tool-call cap applied equally to them.
- **Task set** — the MoE engineering tasks selected from the Task library below.
- **Evaluation budget** — number of runs, GPU-hours, and wall-clock time.

## Inputs

- A research question (e.g. "Do MoE skills improve agent architecture-design
  quality?").
- Task prompts (from the Task library below, possibly adapted to the model).
- An evaluation budget.

## Workflow

1. Define the research question and hypotheses (H0 plus the mechanism hypotheses H1-H5).
2. Select tasks from the Task library (section below), including at least one task that requires arbitrating conflicting guidance and one that requires extrapolating past published recipes — those discriminate skills from retrieval.
3. Define the arms: SAME model, SAME hardware, SAME task set, SAME seeds — arms differ only in retrieval access and skills loaded.
4. Define metrics per task: quality rubrics, cost instrumentation, and reliability measures.
5. Run a headroom check — if the search-enabled arm already saturates a rubric, that task cannot detect an effect; fix or drop it before the full wave.
6. Define and execute the run protocol (n repetitions, ordering, seeding, query logging) — save all outputs and search-query logs under `benchmarks/baseline/` (A0, A1) and `benchmarks/with-skills/` (A2, A3), one subdirectory per arm.
7. Statistical analysis: paired comparison on the primary endpoint, secondaries with multiplicity correction, effect size, mean ± 95% CI.
8. Write the report into `benchmarks/results/`, including the retrieval-only gain (A1 - A0).

## Experimental arms

The baseline is **search-enabled**. Comparing against an agent denied all
information access measures access to information, not the value of skills, and
inflates the effect.

| Arm | Retrieval | Skills | Role |
|-----|-----------|--------|------|
| A0 | No | No | Floor; intrinsic task difficulty |
| A1 | Yes | No | The real control |
| A2 | No | Yes | Skills isolated from retrieval; air-gapped-cluster case |
| A3 | Yes | Yes | Deployed configuration |

Primary comparison is **A3 vs A1**. Always also report **A1 - A0**, the
retrieval-only gain, so readers can see how much of the effect search alone
buys.

Confound controls:

- Disable retrieval at the harness level for A0/A2 — never by telling the model
  not to search.
- Apply the same tool-call cap and the same retrieval backend to A1 and A3.
- Log every search query, returned URL, and retrieved snippet; a run without its
  query log is not reproducible and is excluded.
- Cache or snapshot retrieval responses where the backend allows replay;
  otherwise record the run date and state cross-date comparison as a limitation.
- Skill text must never instruct the agent to search, or A2 and A3 stop being
  separable.

## Hypotheses

| ID | Hypothesis | Comparison | Metric |
|----|-----------|-----------|--------|
| H1 | Skills improve procedural task quality beyond retrieval | A3 > A1 | diagnosis accuracy, rubric total |
| H2 | Skills reduce run-to-run variance | Var(A3) < Var(A1) | score SD across seeds |
| H3 | Skills reduce cost at equal-or-better quality | A3 < A1 | tool calls, tokens, wall-clock |
| H4 | Skills reduce internally inconsistent configurations | A3 < A1 | internal-inconsistency rate |
| H5 | Skills reduce numeric errors in artifacts | A3 < A1 | numeric-consistency error rate |

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

### Task 5: Conflicting-guidance resolution

- **Task id / name:** `task5` — Conflicting-guidance resolution
- **Prompt:** "Choose the capacity factor, auxiliary-loss coefficient, and top-k
  for this hardware budget, and justify each choice against the disagreement in
  the published literature."
- **Inputs:** a hardware budget and a target model size.
- **Outputs:** a decision document (`task5_runM.md`) stating each value, the
  conflicting positions it was chosen against, and the conditions under which
  the choice should be revisited.
- **Metrics:** internal consistency, justification quality, whether stated
  deviation conditions are present.
- **Scoring rubric:** internal consistency is checked against the pre-registered
  incompatible-pair list (deterministic); justification is rubric-scored 0-5.
- **Why this task exists:** retrieval surfaces all the conflicting positions and
  arbitrates none of them. This is where a committed, defended default should
  separate the skills arm from the search arm.

### Task 6: Constrained-hardware design

- **Task id / name:** `task6` — Constrained-hardware design
- **Prompt:** "Design an MoE model under this constraint" — a constraint no
  published recipe matches directly (e.g. limited interconnect bandwidth, a
  fixed expert-parallel degree).
- **Inputs:** the constraint, plus a target dense-equivalent size.
- **Outputs:** an architecture document (`task6_runM.md`).
- **Metrics:** correctness (parameter math), constraint satisfaction,
  expert-utilization awareness.
- **Scoring rubric:** constraint satisfaction is a deterministic check against
  the stated numeric constraint; the rest reuses the Task 1 criteria.
- **Why this task exists:** it forces extrapolation past any single published
  recipe, so a retrieved recipe cannot be copied wholesale.

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

### Cost metrics (recorded on every run, every arm)

- **Tool calls** — total, and search calls specifically.
- **Total tokens** — prompt + completion, including retrieved content pulled
  into context. Retrieval that must be re-paid on every run shows up here.
- **Wall-clock to first valid artifact** — time to an output passing the task's
  validity check, independent of its rubric score.

### Reliability metrics (across seeds within an arm)

- **Score SD** per task per arm — the H2 endpoint.
- **Numeric-consistency error rate** — fraction of artifacts whose stated
  numbers do not close under the task's own formulas (total parameters vs
  expert count × expert size; memory estimate vs stated batch geometry).
  Checked mechanically, never by reading the prose.
- **Internal-inconsistency rate** — fraction of artifacts combining mutually
  incompatible recommendations (e.g. top-1 routing paired with a capacity factor
  justified for top-2), checked against a pre-registered incompatible-pair list.

## Statistical design

- **Sample size:** n runs per arm per task — minimum 5; more with budget. Four
  arms × six tasks × n=5 is 120 runs.
- **Design:** paired across arms by (task, seed); the unit of analysis is the
  per-pair difference.
- **Primary endpoint:** A3 vs A1 on task quality — one pre-registered test.
- **Multiplicity:** every other comparison is secondary, reported with a Holm
  correction over the pre-registered secondary family, and never presented as
  confirmatory.
- **Effect size:** Cohen's d on the paired differences.
- **Variance endpoint:** SD ratio between arms with bootstrap CIs — do not use a
  normal-theory variance ratio test on rubric scores.
- **Reporting:** mean ± 95% CI per metric per arm, plus the paired difference
  and its CI, plus the retrieval-only gain (A1 - A0).
- **Pre-registration:** hypotheses, rubrics, thresholds, the incompatible-pair
  list, and exclusion rules are fixed before any scoring.

## Anti-patterns

- **Denying the baseline web search.** The single most effect-inflating mistake
  available here: it measures information access, not skills. The control arm
  gets retrieval.
- **Disabling search by prompt instruction** rather than by unregistering the
  tool — the model may search anyway, or may behave differently because it was
  told not to. Contaminates the arm either way.
- Running search-enabled arms without logging queries (irreproducible).
- Reporting the skill effect without reporting the retrieval-only gain (A1 - A0).
- Comparing different models (all arms must be the SAME model).
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

| Task | Metric | A0 bare | A1 search | A2 skills | A3 skills+search | A3-A1 | Cohen's d |
|------|--------|---------|-----------|-----------|------------------|-------|-----------|
| task1 | correctness (total /25) |  |  |  |  |  |  |
| task2 | successful launch |  |  |  |  |  |  |
| task3 | accuracy (0-1) |  |  |  |  |  |  |
| task4 | throughput (tok/s) |  |  |  |  |  |  |
| task5 | internal consistency |  |  |  |  |  |  |
| task6 | constraint satisfaction |  |  |  |  |  |  |
| all | score SD (H2) |  |  |  |  |  |  |
| all | total tokens (H3) |  |  |  |  |  |  |
| all | numeric-error rate (H5) |  |  |  |  |  |  |

3. **Analysis plan** — how to interpret the results, what "significant" means,
   and how failures are reported. A tie on quality against the search-enabled
   arm, with lower variance and lower cost, is a legitimate positive result and
   should be reported as exactly that — not dressed up as a quality win.

## Evaluation criteria

- **Reproducibility:** exact commands, seeds, environment, and search-query logs
  recorded for every run.
- **Strong baseline:** the control arm has retrieval; the study never claims
  credit for information the baseline was simply denied.
- **Controlled comparison:** arms differ ONLY in retrieval access and skills
  loaded.
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
- `benchmarks/baseline/` — arms A0 (bare) and A1 (search-enabled control), one
  subdirectory per arm, each with its search-query logs.
- `benchmarks/with-skills/` — arms A2 (skills, offline) and A3 (skills+search).
- `benchmarks/results/` — aggregated results tables and analysis reports.
