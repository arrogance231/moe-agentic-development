# Orchestrator (planned — Phase 1 placeholder)

## Purpose

The orchestrator coordinates the MoE skills to complete an end-to-end MoE
engineering task. It sequences the workflow across skills — architecture →
training → debugging → performance — and delegates sub-tasks to the appropriate
skill, stitching the results into a single coherent deliverable.

## Inputs

- The task prompt.
- Model and hardware constraints (dense-equivalent size, GPU count, memory
  budget, target throughput).

## Outputs

- The completed task artifacts (architecture document, training config,
  diagnosis, optimization report) produced by the delegated skills.

## Status

Placeholder — implementation planned in Phase 4.
