# Benchmark Protocol: Does the MoE Skill Suite Improve Agent Performance?

This is a complete, ready-to-adapt protocol for the 4-task MoE engineering
benchmark. Copy it into `benchmarks/results/`, fill in the concrete model and
hardware, and execute it.

## Research question and hypotheses

- **Research question:** Do the MoE skills improve agent performance on MoE
  engineering tasks vs the same agent without skills?
- **H0 (null):** Agents with the MoE skills loaded perform no better than the
  same agents without them on MoE engineering tasks (mean paired difference =
  0).
- **H1 (alternative):** Agents with the MoE skills loaded outperform the same
  agents without them (mean paired difference > 0).

## Design

- **Baseline:** model X (specify checkpoint/version), no skills loaded.
- **Treatment:** model X + 5 MoE skills, loaded via the skill-loading
  mechanism.
- **Design type:** paired — the same 4 tasks, the same seeds, run under both
  conditions.
- **Sample size:** n = 5 per condition per task (increase with budget).

## Tasks

The four tasks are taken verbatim from the SKILL.md Task library.

1. **Task 1: Architecture Design** — "Design a 1B dense-equivalent MoE model."
   Produces a structured architecture document.
2. **Task 2: Training Setup** — "Create a training configuration for this
   architecture." Produces a launchable training config for the Task 1 design.
3. **Task 3: Debugging** — given a broken run (expert collapse, NaN loss, OOM,
   or comm bottleneck), "Diagnose the failure and propose a fix." Produces a
   structured diagnosis.
4. **Task 4: Optimization** — given an existing run, "Improve throughput."
   Produces an optimized config with before/after metrics.

Each task provides the agent with its inputs (see Task library) and expects the
documented output format.

## Metrics and rubrics

| Task | Metric | Scorer | Scale |
|------|--------|--------|-------|
| task1 | correctness (param math) | `evaluators/score_architecture.py` | 0-5 |
| task1 | efficiency | `evaluators/score_architecture.py` | 0-5 |
| task1 | expert utilization of proposal | `evaluators/score_architecture.py` | 0-5 |
| task1 | completeness | `evaluators/score_architecture.py` | 0-5 |
| task1 | justification | `evaluators/score_architecture.py` | 0-5 |
| task1 | total | `evaluators/score_architecture.py` | /25, PASS ≥ 15 |
| task2 | successful launch | manual binary check (pre-recorded) | 0/1 |
| task2 | memory efficiency | numeric threshold | ratio |
| task2 | throughput | numeric threshold | tokens/sec |
| task3 | diagnosis accuracy | `evaluators/score_debugging.py` | 0-1 weighted |
| task4 | throughput improvement | numeric delta | ratio |
| task4 | GPU utilization / memory | numeric delta | % / bytes |

Rubrics are pre-registered: `score_architecture.py` and `score_debugging.py`
are deterministic, and the manual launch check is recorded before unblinding.
See `knowledge/methodology.md` for the full rubric rules.

## Run protocol

1. Record the environment: model X version, hardware, software stack, and the
   skill-loading mechanism.
2. For each seed s in {seed1..seed5} and each task t in {task1..task4}:
   - **Baseline:** invoke the agent WITHOUT skills:
     `agent run --model "model X" --prompt <task t prompt> --seed <s>`
   - **Treatment:** invoke the same agent WITH the MoE skills loaded (load
     `skills/` via the runtime's skill flag/environment) using the same prompt
     and seed:
     `agent run --model "model X" --skills skills/ --prompt <task t prompt> --seed <s>`
3. Save every raw output under the benchmark directories:
   - `benchmarks/baseline/taskN_runM.md` (plus `task2_runM.json` configs,
     `task3_runM.json` diagnoses)
   - `benchmarks/with-skills/taskN_runM.md` (plus the same extras)
4. Score each output:
   - task1 and task3 outputs with the evaluator scripts:
     `python evaluators/score_architecture.py --proposal <file> --json`
     `python evaluators/score_debugging.py --diagnosis <file> --ground-truth benchmarks/results/gt_task3.json --json`
   - task2/task4 numeric metrics with the pre-registered formulas; the launch
     check is recorded by a human before unblinding.
5. Store per-run scores next to the raw outputs.

## Statistical analysis plan

- Unit of analysis: the paired difference (treatment - baseline) for each
  (task, seed) pair.
- Report mean ± 95% CI per condition and per task, plus the mean paired
  difference and its CI.
- Report Cohen's d on the paired differences.
- Test H0 with a paired t-test (or Wilcoxon signed-rank if the differences are
  strongly non-normal), significance level 0.05, pre-registered.
- Failures (launch failures, invalid outputs, OOMs) are recorded and reported;
  handle them per the rule fixed before scoring.

## Results template

| Task | Metric | Baseline mean ± CI | Treatment mean ± CI | Delta | Effect size (Cohen's d) |
|------|--------|--------------------|---------------------|-------|--------------------------|
| task1 | correctness (total /25) |  |  |  |  |
| task2 | successful launch |  |  |  |  |
| task2 | memory efficiency |  |  |  |  |
| task2 | throughput (tok/s) |  |  |  |  |
| task3 | accuracy (0-1) |  |  |  |  |
| task4 | throughput (tok/s) |  |  |  |  |
| task4 | GPU utilization (%) |  |  |  |  |
