# benchmarks/baseline

Outputs for the two no-skills arms, one subdirectory per arm:

- `A0/` — bare model: no retrieval, no skills. The floor condition.
- `A1/` — **the real control**: retrieval enabled, no skills. A competent
  engineer-agent that looks things up.

Layout per arm: `taskN_runM.md` (plus `task2_runM.json` configs and
`task3_runM.json` diagnoses) with per-run scores in `taskN_runM.metrics.json`.

A1 runs additionally carry `taskN_runM.queries.jsonl` — every search query, the
URLs returned, and the retrieved snippets. Retrieval drifts across days, so an
A1 run without its query log is not reproducible and is excluded from analysis.

Populated by the `moe-benchmarking` skill.
