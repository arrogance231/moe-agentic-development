# Whitepaper Handoff

This document defines what a benchmark-execution run must produce so the
outputs can be handed to a separate Claude session for whitepaper writing,
without that session needing to re-derive anything from raw logs.

## Required artifacts per run

Every execution pass against the AMD environment must produce, under
`benchmarks/results/<run-id>/`:

1. **`environment.md`** — GPU model(s), count, VRAM, ROCm version, driver
   version, kernel version, PyTorch/framework build, interconnect topology
   (`rocm-smi`, `rocminfo`, `amd-smi` output captured verbatim, not
   summarized). Required because BENCHMARK.md's "same hardware" precondition
   means results from a different environment can never be pooled with these.
2. **`run-log.md`** — a running, timestamped log of what was executed, in the
   order it happened: setup steps, each arm/task/seed invocation, and outcome.
   Append-only during the run, not reconstructed afterward.
3. **`failures.md`** — every failure (OOM, launch failure, crash, hang, invalid
   output, retrieval timeout), with: what was attempted, the exact error,
   suspected cause, and whether/how it was resolved or excluded. Honest
   reporting per BENCHMARK.md — failures are findings, not noise to omit.
4. **`results-<task>.md`** — one file per task, containing the raw scored
   outputs and the per-arm table from BENCHMARK.md's results template, filled
   in with real numbers.
5. **`summary.md`** — aggregated results table across all arms/tasks, the
   retrieval-only gain (A1 − A0), the primary endpoint (A3 vs A1) with effect
   size, and a plain-language statement of what was and wasn't supported by
   the data. No interpretation beyond what the numbers show.
6. **`visualizations/`** — chart images (or a Claude Artifact link recorded in
   `summary.md`) for: per-arm score distributions, cost (tokens/tool
   calls/wall-clock) by arm, and score variance by arm. Charts, not just
   tables — the whitepaper session needs figures it can drop in directly.

## What the whitepaper session should NOT have to do

- Re-run any experiment or re-derive numbers from logs.
- Guess at hardware specs or environment details.
- Reconcile conflicting run logs — if a run was excluded, `failures.md` says
  why before the whitepaper session ever sees the data.

## Handoff checklist

Before treating a run as ready to hand off:

- [ ] `environment.md` has raw tool output, not a paraphrase.
- [ ] `run-log.md` covers the full run, timestamped, no gaps.
- [ ] `failures.md` exists even if empty (state "no failures observed").
- [ ] Every task in the BENCHMARK.md task library has a `results-<task>.md`.
- [ ] `summary.md` states the retrieval-only gain before the skill effect.
- [ ] At least one chart exists per cost/quality/variance dimension.
