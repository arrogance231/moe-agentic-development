# Summary — run-20260808-0559

Full four-arm (A0/A1/A2/A3) x six-task benchmark, n=5 seeds per arm per task, 120/120 runs completed (5 transient errors, all resolved via retry, zero permanent exclusions). Model under test: OpenCode CLI driving `opencode/deepseek-v4-flash-free` via the OpenCode Zen API. Retrieval backend: keyless DuckDuckGo Lite HTML scrape (no paid search API available) — logged as a validity limitation (rate limits, less reliable structured results than a paid backend such as Tavily/Brave).

## Sample size caveat

n=5 paired runs per arm per task is the BENCHMARK.md-specified minimum. All effect sizes and p-values below should be read as low-powered point estimates, not confirmatory results. This matches the design's stated intent to power variance (H2) and cost (H3) endpoints fully while treating quality endpoints (H1) as directional at this n.

## Retrieval-only gain (A1 − A0) — reported before any skill effect

Per BENCHMARK.md's design intent, this is the number a skeptical reader looks for first: how much of any observed effect is available from search alone, with no skills involved.

| Task | Metric | A0 mean | A1 mean | A1−A0 | Cohen's d |
|---|---|---|---|---|---|
| task1 | correctness (total /25) | 21 | 21 | 0 | 0.0 |
| task2 | successful launch | 1 | 1 | 0 | n/a |
| task3 | accuracy (0-1) | 0.463 | 0.44 | -0.023 | -0.92 |
| task4 | throughput delta (frac) | 1.026 | 0.575 | -0.452 | n/a |
| task5 | internal consistency | 0 | 0.2 | 0.2 | 0.447 |
| task6 | constraint satisfaction + arch total | 23.4 | 23.8 | 0.4 | 0.447 |

## Primary endpoint: A3 vs A1 (pre-registered)

| Task | Metric | A1 mean | A3 mean | A3−A1 | Cohen's d | paired t p |
|---|---|---|---|---|---|---|
| task1 | correctness (total /25) | 21 | 22.2 | 1.2 | 0.447 | 0.3739 |
| task2 | successful launch | 1 | 1 | 0 | n/a | n/a |
| task3 | accuracy (0-1) | 0.44 | 0.474 | 0.034 | 1.601 | 0.0232 |
| task4 | throughput delta (frac) | 0.575 | 0.36 | -0.215 | n/a | n/a |
| task5 | internal consistency | 0.2 | 0 | -0.2 | -0.447 | 0.3739 |
| task6 | constraint satisfaction + arch total | 23.8 | 24.2 | 0.4 | 0.239 | 0.6213 |

**Task1 ceiling-effect caveat carried forward:** the A1-only headroom check found task1 scores near the rubric max (21/25, 25/25 at n=2). The full-wave task1 numbers above should be read with reduced power to detect an A3-vs-A1 difference for that reason, consistent with methodology.md's pre-registered handling of ceiling effects (report the saturation, do not force a null result to look like evidence of no effect).

## Secondary comparisons (exploratory, Holm-corrected)

Per methodology.md, all comparisons other than A3 vs A1 are secondary and reported as exploratory, not confirmatory. Holm correction applied across the full secondary family below.

| Comparison | mean diff | raw p | Holm-adjusted p |
|---|---|---|---|
| task1:A1-A0 | 0 | 1.0 | 1.0 |
| task2:A1-A0 | 0 | n/a | n/a |
| task3:A1-A0 | -0.023 | 0.1087 | 1.0 |
| task4:A1-A0 | -0.452 | n/a | n/a |
| task5:A1-A0 | 0.2 | 0.3739 | 1.0 |
| task6:A1-A0 | 0.4 | 0.3739 | 1.0 |
| task1:A2-A0 | 1.6 | 0.405 | 1.0 |
| task1:A2vA1 | 1.6 | 0.3375 | 1.0 |
| task1:A3-A2 | -0.4 | 0.8541 | 1.0 |
| task2:A2-A0 | 0 | n/a | n/a |
| task2:A2vA1 | 0 | n/a | n/a |
| task2:A3-A2 | 0 | n/a | n/a |
| task3:A2-A0 | -0.024 | 0.0479 | 0.6709 |
| task3:A2vA1 | -0.001 | 0.9285 | 1.0 |
| task3:A3-A2 | 0.035 | 0.0434 | 0.6507 |
| task4:A2-A0 | -0.604 | n/a | n/a |
| task4:A2vA1 | -0.152 | n/a | n/a |
| task4:A3-A2 | -0.063 | 0.4499 | 1.0 |
| task5:A2-A0 | 0 | n/a | n/a |
| task5:A2vA1 | -0.2 | 0.3739 | 1.0 |
| task5:A3-A2 | 0 | n/a | n/a |
| task6:A2-A0 | 1.2 | 0.0705 | 0.9163 |
| task6:A2vA1 | 0.8 | 0.1778 | 1.0 |
| task6:A3-A2 | -0.4 | 0.6213 | 1.0 |

## Reliability endpoints (H2, H5)

| Task | SD(A0) | SD(A1) | SD(A2) | SD(A3) |
|---|---|---|---|---|
| task1 | 2.449 | 1.414 | 2.191 | 2.683 |
| task2 | 0.0 | 0.0 | 0.0 | 0.0 |
| task3 | 0.011 | 0.024 | 0.015 | 0.018 |
| task4 | 0.42 | 0.205 | 0.19 | 0.061 |
| task5 | 0.0 | 0.447 | 0.0 | 0.0 |
| task6 | 0.894 | 1.095 | 0.894 | 1.095 |

## Cost endpoints (H3) — aggregated across all 6 tasks

| Arm | mean tokens/run | mean tool calls/run | mean wall-clock (s)/run |
|---|---|---|---|
| A0 | 27459.2 | 1.77 | 74.2 |
| A1 | 41592.9 | 2.7 | 87.4 |
| A2 | 52278.4 | 2.03 | 90.3 |
| A3 | 109937.9 | 3.33 | 109.8 |

## What is and isn't supported by this data

- Retrieval-only gain (A1−A0) is reported above per-task, before any skill
  effect; it is the honest floor of "how much does search alone buy you."
- The pre-registered primary endpoint (A3 vs A1) is reported per task with
  Cohen's d and a paired t-test p-value at n=5 — these are point estimates
  from a small sample and are not strong enough to claim statistical
  significance on their own for any single task; treat directionality
  (sign and rough magnitude of A3−A1) as the informative signal at this n,
  not the p-value threshold.
- Task1 has a documented ceiling effect (near-max scores in the headroom
  check); any near-zero A3−A1 difference there is consistent with "no
  headroom to detect an effect," not necessarily "skills add nothing."
- Task2/task4 numeric fields (memory efficiency, throughput, GPU utilization)
  are self-reported by the model under test — no real training job executes
  in this benchmark. These are internal-consistency/plausibility signals,
  not measured ground truth. This is a real limitation of the benchmark as
  currently specified in BENCHMARK.md (which does not call for an actual
  training run), stated here rather than treated as a hard result.
- Secondary comparisons (A2−A0, A2 vs A1, A3−A2) are exploratory and
  Holm-corrected; none should be read as confirmatory on their own.
- Retrieval used a keyless, unpaid backend (DuckDuckGo Lite scrape) rather
  than the paid API BENCHMARK.md's design implicitly assumes — a stated
  limitation on data quality/reliability for A1/A3, not a silent gap.
- GPU utilization telemetry (rocm-smi/rocminfo/amd-smi) was unavailable on
  this VM despite a physical MI300X being present (see environment.md);
  this does not affect the benchmark itself since the model under test runs
  via API, not local GPU inference, but it means no independently measured
  GPU utilization numbers exist for any arm — only the model's own claims
  in task4 outputs.


## Interactive dashboard

Published Claude Artifact (headline result, retrieval-only gain, primary endpoint, variance, cost — same data as this document): https://claude.ai/code/artifact/69695947-9e4c-4cf2-a0c4-464c8e938cd0

## Handoff checklist (docs/whitepaper-handoff.md)

- [x] environment.md has raw tool output, not a paraphrase — verbatim rocm-smi/rocminfo/amd-smi (all absent, logged as such)/uname/lscpu/lspci/free/df output captured; ROCm tooling absence noted as non-blocking since the model under test runs via API, not local GPU inference.
- [x] run-log.md covers the full run, timestamped, no gaps — append-only from session start through the last of 120 wave runs and scoring phase (29.7KB, continuous timestamps).
- [x] failures.md exists even if empty — not empty: documents the ROCm/pip3 setup gaps, the contamination bug found and fixed during the headroom check, and every transient wave-run retry.
- [x] Every task in the BENCHMARK.md task library has a results-<task>.md — all 6 present (results-task1.md .. results-task6.md).
- [x] summary.md states the retrieval-only gain before the skill effect — "Retrieval-only gain (A1-A0)" section appears before "Primary endpoint: A3 vs A1" section, in that order.
- [x] At least one chart exists per cost/quality/variance dimension — visualizations/score_distributions.png (quality), visualizations/cost_by_arm.png (cost), visualizations/score_variance_by_arm.png (variance), plus the published Artifact dashboard covering all three.

All six items pass. This run is ready to hand off to a whitepaper-writing session without further raw-log derivation.
