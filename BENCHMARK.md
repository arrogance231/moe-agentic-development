# Benchmark

This document summarizes the benchmark methodology for measuring whether the MoE
skills measurably improve agent performance on MoE engineering tasks. The
experiments are controlled agent experiments: the same model, hardware, and task
set run with and without the skill suite loaded, and the outputs are scored
against pre-registered rubrics.

## Research question and design

**Research question:** Do domain-specific skills measurably improve MoE
engineering performance vs a general LLM?

**Design:** a paired, controlled comparison.

- **Baseline condition:** the agent WITHOUT MoE skills.
- **Treatment condition:** the SAME agent with the 5-skill MoE suite loaded
  (moe-architecture, moe-training, moe-debugging, moe-performance,
  moe-benchmarking).
- **Controls:** same model, same hardware, same task set; baseline and treatment
  run on identical tasks and seeds so each pair differs only in whether skills
  are loaded.
- **Paired runs:** every baseline run has a matched treatment run.

## Tasks

| Task | Prompt | Metrics |
|------|--------|---------|
| Task 1: Architecture Design | "Design a 1B dense-equivalent MoE model" | correctness, efficiency, expert utilization |
| Task 2: Training Setup | Create a training config consistent with the architecture | successful launch, memory efficiency, throughput |
| Task 3: Debugging | Diagnose broken runs (expert collapse, NaN, OOM, comm bottleneck) | diagnosis accuracy, fix success, time saved |
| Task 4: Optimization | Improve throughput of a working run | before/after tokens/sec, GPU util, memory |

## Metrics and scoring

Metrics and scoring rubrics are defined per task in the `moe-benchmarking`
skill, which owns the methodology:

- Rubric checklists: see `skills/moe-benchmarking/SKILL.md` (Task library and
  "Metrics definitions and scoring rubrics").
- Evaluator scripts: `skills/moe-benchmarking/evaluators/score_architecture.py`
  and `skills/moe-benchmarking/evaluators/score_debugging.py` score outputs
  deterministically so scoring cannot be influenced by reviewer expectations.

Rubrics are pre-registered: they are fixed before scoring begins and are never
re-tuned to fit results.

## Statistical design

- **Sample size:** n ≥ 5 runs per condition per task (more with budget).
- **Comparison:** paired comparison — each baseline/treatment pair shares the
  same task and seed.
- **Effect size:** Cohen's d on the paired differences.
- **Reporting:** mean ± 95% CI per metric per condition.
- **Pre-registration:** rubrics and thresholds are fixed before scoring.

## Results table template

| Task | Metric | Baseline mean ± CI | With-skills mean ± CI | Δ | Effect size |
|------|--------|--------------------|------------------------|----|-------------|
| task1 | correctness (total /25) | | | | |
| task2 | successful launch | | | | |
| task3 | accuracy (0-1) | | | | |
| task4 | throughput (tok/s) | | | | |

## Running it

The run protocol, scoring, and analysis are executed by the `moe-benchmarking`
skill (see `skills/moe-benchmarking/SKILL.md` and
`skills/moe-benchmarking/examples/benchmark-protocol.md`). Raw outputs land in
`benchmarks/baseline/` (agent WITHOUT skills) and `benchmarks/with-skills/`
(agent WITH skills); aggregated results and analysis go in `benchmarks/results/`.
