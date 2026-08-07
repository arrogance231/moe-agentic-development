# Experiment-manager (planned — Phase 1 placeholder)

## Purpose

The experiment-manager runs benchmark experiments using the `moe-benchmarking`
skill. It manages baseline and treatment runs (agent without vs with MoE skills),
collects raw outputs, invokes the evaluator scripts, and aggregates the scored
results into the benchmark results table.

## Inputs

- The experiment protocol (tasks, metrics, rubrics, run ordering, seeds).
- The evaluation budget (number of runs, GPU-hours, wall-clock time).

## Outputs

- The aggregated results table (baseline vs treatment per task/metric).
- The analysis report (statistical analysis, effect sizes, honest reporting of
  failures).

## Status

Placeholder — implementation planned in Phase 4.
