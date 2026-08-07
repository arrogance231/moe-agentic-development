# Benchmark Protocol: Do MoE Skills Beat a Search-Enabled Agent?

This is a complete, ready-to-adapt protocol for the 6-task MoE engineering
benchmark. Copy it into `benchmarks/results/`, fill in the concrete model,
hardware, and retrieval backend, and execute it.

## Research question and hypotheses

- **Research question:** Given an agent that can already retrieve MoE knowledge
  on demand, does packaging that domain as procedural skills produce measurably
  better, cheaper, and more reproducible MoE engineering work than retrieval
  alone?
- **H0 (null):** Loading the MoE skills changes nothing against a
  search-enabled baseline, on any endpoint (mean paired difference = 0).
- **H1:** Skills improve procedural task quality beyond retrieval (A3 > A1 on
  task quality). **Primary endpoint.**
- **H2:** Skills reduce run-to-run variance (SD(A3) < SD(A1)).
- **H3:** Skills reduce cost at equal-or-better quality (fewer tool calls,
  tokens, and less wall-clock in A3 than A1).
- **H4:** Skills reduce internally inconsistent configurations.
- **H5:** Skills reduce numeric errors in produced artifacts.

The baseline is deliberately strong. An agent denied web search would be easy to
beat, and beating it would demonstrate only that MoE knowledge helps — which is
not in question.

## Design: four arms

| Arm | Retrieval | Skills | Role |
|-----|-----------|--------|------|
| A0 | No | No | Floor |
| A1 | Yes | No | Control |
| A2 | No | Yes | Skills isolated; air-gapped-cluster case |
| A3 | Yes | Yes | Deployed configuration |

- **Model:** model X (specify checkpoint/version), identical across arms.
- **Retrieval backend:** specify the search/doc tool and its version, shared by
  A1 and A3, with an identical per-run tool-call cap.
- **Design type:** paired — the same 6 tasks and the same seeds across all arms.
- **Sample size:** n = 5 per arm per task (increase with budget) — 120 runs.
- **Primary comparison:** A3 vs A1. **Always also report A1 - A0**, the
  retrieval-only gain.

## Tasks

Taken verbatim from the SKILL.md Task library.

1. **Task 1: Architecture Design** — "Design a 1B dense-equivalent MoE model."
2. **Task 2: Training Setup** — "Create a training configuration for this
   architecture."
3. **Task 3: Debugging** — given a broken run (expert collapse, NaN loss, OOM,
   or comm bottleneck), "Diagnose the failure and propose a fix."
4. **Task 4: Optimization** — given an existing run, "Improve throughput."
5. **Task 5: Conflicting-guidance resolution** — choose capacity factor,
   aux-loss coefficient, and top-k for a stated hardware budget and justify each
   against the disagreement in the literature.
6. **Task 6: Constrained-hardware design** — design under a constraint no
   published recipe matches directly.

Tasks 5 and 6 carry the discriminating power of the study: 1-4 are partly
solvable by retrieving a published recipe, while 5 forces arbitration between
conflicting sources and 6 forces extrapolation past all of them.

**Headroom check (run first).** Execute n = 2 of arm A1 on tasks 1 and 2. If A1
already scores at or near the rubric maximum, the task has no headroom and
cannot detect an effect — fix or drop it before spending the full budget.

## Metrics and rubrics

| Task | Metric | Scorer | Scale |
|------|--------|--------|-------|
| task1 | correctness (param math) | `evaluators/score_architecture.py` | 0-5 |
| task1 | efficiency | `evaluators/score_architecture.py` | 0-5 |
| task1 | expert utilization of proposal | `evaluators/score_architecture.py` | 0-5 |
| task1 | completeness | `evaluators/score_architecture.py` | 0-5 |
| task1 | justification | `evaluators/score_architecture.py` | 0-5 |
| task1 | total | `evaluators/score_architecture.py` | /25, PASS >= 15 |
| task2 | successful launch | manual binary check (pre-recorded) | 0/1 |
| task2 | memory efficiency | numeric threshold | ratio |
| task2 | throughput | numeric threshold | tokens/sec |
| task3 | diagnosis accuracy | `evaluators/score_debugging.py` | 0-1 weighted |
| task4 | throughput improvement | numeric delta | ratio |
| task4 | GPU utilization / memory | numeric delta | % / bytes |
| task5 | internal consistency | incompatible-pair list | 0/1 |
| task5 | justification quality | rubric | 0-5 |
| task6 | constraint satisfaction | deterministic numeric check | 0/1 |
| task6 | correctness (param math) | `evaluators/score_architecture.py` | 0-5 |

Recorded on **every** run in every arm:

| Metric | Source | Hypothesis |
|--------|--------|-----------|
| tool calls (total, and search calls) | harness log | H3 |
| total tokens (prompt + completion, incl. retrieved content) | harness log | H3 |
| wall-clock to first valid artifact | harness log | H3 |
| numeric-consistency errors | mechanical check on the artifact | H5 |
| internal inconsistencies | incompatible-pair list | H4 |

Rubrics are pre-registered: `score_architecture.py` and `score_debugging.py`
are deterministic, and the manual launch check is recorded before unblinding.
The incompatible-pair list is fixed before any run. See
`knowledge/methodology.md` for the full rubric rules and the circularity check.

## Run protocol

1. Record the environment: model X version, hardware, software stack, the
   skill-loading mechanism, the retrieval backend and its tool-call cap, and the
   run date (retrieval results drift).
2. For each seed s in {seed1..seed5} and each task t in {task1..task6}, run all
   four arms with a byte-identical prompt:
   - **A0:** `agent run --model "model X" --no-tools-search --prompt <task t> --seed <s>`
   - **A1:** `agent run --model "model X" --tools search --prompt <task t> --seed <s>`
   - **A2:** `agent run --model "model X" --no-tools-search --skills skills/ --prompt <task t> --seed <s>`
   - **A3:** `agent run --model "model X" --tools search --skills skills/ --prompt <task t> --seed <s>`

   Retrieval must be disabled by **not registering the tool** in A0/A2, never by
   instructing the model not to search.
3. Save every raw output and the per-run search-query log:
   - `benchmarks/baseline/A0/taskN_runM.md`, `benchmarks/baseline/A1/taskN_runM.md`
     (plus `task2_runM.json` configs, `task3_runM.json` diagnoses, and
     `taskN_runM.queries.jsonl` for A1)
   - `benchmarks/with-skills/A2/...`, `benchmarks/with-skills/A3/...` (same
     layout; query logs for A3)

   A search-enabled run with no query log is not reproducible: exclude it and
   record the exclusion.
4. Score each output:
   - task1, task6 with `python evaluators/score_architecture.py --proposal <file> --json`
   - task3 with `python evaluators/score_debugging.py --diagnosis <file> --ground-truth benchmarks/results/gt_task3.json --json`
   - task2/task4/task5 numeric and consistency metrics with the pre-registered
     formulas and the incompatible-pair list; the launch check is recorded by a
     human before unblinding.
5. Store per-run scores and cost metrics next to the raw outputs as
   `taskN_runM.metrics.json`.

## Statistical analysis plan

- Unit of analysis: the paired difference for each (task, seed) pair across arms.
- **Primary:** A3 - A1 on task quality, paired t-test (or Wilcoxon signed-rank
  if the differences are strongly non-normal), alpha = 0.05, pre-registered.
- **Secondary family** (Holm-corrected, reported as exploratory): A1 - A0,
  A2 - A0, A2 vs A1, A3 - A2, and the interaction (A3 - A1) - (A2 - A0).
- Variance endpoint (H2): SD per arm per task with bootstrap CIs on the SD ratio.
- Cost endpoints (H3): mean +- 95% CI per arm on tool calls, tokens, wall-clock.
- Report Cohen's d on the paired differences for every reported comparison.
- Failures (launch failures, invalid outputs, OOMs, missing query logs) are
  recorded and reported; handle them per the rule fixed before scoring.
- **Report A1 - A0 before reporting the skill effect.** It states how much of
  the total gain search alone already provides.

## Results template

| Task | Metric | A0 bare | A1 search | A2 skills | A3 skills+search | A3-A1 | Cohen's d |
|------|--------|---------|-----------|-----------|------------------|-------|-----------|
| task1 | correctness (total /25) |  |  |  |  |  |  |
| task2 | successful launch |  |  |  |  |  |  |
| task2 | memory efficiency |  |  |  |  |  |  |
| task2 | throughput (tok/s) |  |  |  |  |  |  |
| task3 | accuracy (0-1) |  |  |  |  |  |  |
| task4 | throughput (tok/s) |  |  |  |  |  |  |
| task4 | GPU utilization (%) |  |  |  |  |  |  |
| task5 | internal consistency |  |  |  |  |  |  |
| task6 | constraint satisfaction |  |  |  |  |  |  |
| all | score SD (H2) |  |  |  |  |  |  |
| all | tool calls (H3) |  |  |  |  |  |  |
| all | total tokens (H3) |  |  |  |  |  |  |
| all | wall-clock to first artifact (H3) |  |  |  |  |  |  |
| all | numeric-error rate (H5) |  |  |  |  |  |  |

## Interpreting the outcome

- **A3 > A1 on quality:** skills add something retrieval cannot supply. Report
  which tasks carry the effect — expect 5 and 6 to dominate.
- **A3 ~ A1 on quality but lower variance and cost:** the most likely honest
  outcome, and a real result. Report it as reproducibility and efficiency, not
  as a quality win.
- **A3 ~ A1 everywhere:** skills are redundant with retrieval for these tasks.
  Publish that. It is a genuine finding about the limits of skill packaging, and
  the correct response is not to re-tune the rubric.
- **A2 ~ A1:** skills substitute for search, which matters directly for
  egress-restricted training clusters.
