# Roadmap

Current phase: **Phase 1 — Skill framework** (active). Phase 0 is complete;
Phases 2-5 are future work.

## Phase 0: Research

**Status:** complete

- [x] Literature review on MoE training and agentic skills
- [x] Identified five skill domains (architecture, training, debugging, performance, benchmarking)
- [x] Validated skill format (SKILL.md frontmatter) against Claude Code / OpenCode

## Phase 1: Skill framework

**Status:** active

- [x] Skill loader + CLI (`moe-skills list/validate/info/deploy`)
- [x] Five skills authored (architecture, training, debugging, performance, benchmarking)
- [x] Skill-spec documentation
- [x] Benchmark harness (evaluators + methodology)
- [x] Four-arm research design with a search-enabled control (see BENCHMARK.md)
- [x] Tasks 5 and 6 (conflicting-guidance resolution, constrained-hardware design)
- [x] Cost instrumentation (tool calls, tokens, wall-clock) in the run harness
- [x] Search-query logging and retrieval caching for arms A1/A3
- [ ] Independent review of rubrics for skill-rubric circularity
- [x] Agent role docs (orchestrator, experiment-manager) — placeholders
- [x] Adapters (HF/DeepSpeed/Megatron) — placeholders
- [x] Running the first benchmark wave (run-20260808-0559, n=5 x 4 arms x 6 tasks, see benchmarks/results/run-20260808-0559/summary.md)
- [ ] Integrating feedback into skills

## Phase 2: Benchmark execution

**Status:** complete (first wave)

- [x] Headroom check: confirm the search-enabled arm (A1) does not saturate the rubrics
      — found a ceiling effect on task1 (scores 21-25/25), proceeded per pre-registered
      decision since tasks 5/6 exist to discriminate where 1-4 saturate.
- [x] Run all four arms (A0 bare, A1 search, A2 skills, A3 skills+search), paired by task and seed
- [x] Statistical analysis: primary endpoint A3 vs A1, secondaries with Holm correction
- [x] Report the retrieval-only gain (A1 − A0) alongside the skill effect
- [x] Publish results in benchmarks/results (run-20260808-0559)
- [ ] Repeat the wave at larger n once real-training validation (Phase 2.5) is in place, to
      check whether task2/task4 conclusions change once they're graded on real outcomes
      instead of self-reported/schema-validated proxies

## Phase 2.5: Real-training validation

**Status:** future — not started

**Why:** the first benchmark wave (run-20260808-0559) scored task2 ("training setup") and
task4 ("optimization") entirely on the *proposal text* — JSON-schema validity and
architecture-consistency for task2, and the model's own claimed throughput delta against a
stated baseline for task4. No config was ever actually launched, no training step ran, and
no throughput/memory number was independently measured. This is a known, explicitly logged
limitation (see `benchmarks/harness/thresholds.md` and `failures.md` in
`benchmarks/results/run-20260808-0559/`). The GPU (MI300X) sat effectively idle during the
whole run since the benchmarked agent (OpenCode + DeepSeek V4 Flash) runs via API, not
locally.

This phase closes that gap by actually training the MoE architectures the agent designs,
so task2/task4 are scored on real outcomes rather than proposal quality.

- [ ] Install the ROCm + PyTorch (or JAX) stack on the GPU VM — absent as of the first run
      (`rocm-smi`/`rocminfo`/`amd-smi`/`torch` all unavailable; MI300X VF confirmed present
      via `lspci` but with no userspace driver stack installed)
- [ ] Build a minimal MoE training script that consumes a task2-style config (the same
      schema the agent already produces) and actually launches training on the MI300X —
      reuse the HF/DeepSpeed/Megatron adapter placeholders in `adapters/` as the starting
      point rather than writing a new trainer from scratch
- [ ] Wire real launch success / OOM / crash into task2 scoring (replace the current
      schema-validity proxy) and real measured throughput/memory into task4 scoring
      (replace the current self-reported-delta proxy), per `benchmarks/harness/thresholds.md`
- [ ] Pilot: run one real training job per arm (4 total) on a representative task2 config,
      for a short validation run only (a few hundred–few thousand steps on a data subset —
      enough to confirm the config launches, capture real throughput/memory, and see loss
      decrease; not a converged model) — decided scope per user 2026-08-08
- [ ] If the pilot works, decide whether to scale to all 20 task2 configs (5 seeds × 4 arms)
      before re-running the full wave
- [ ] Re-run task2/task4 scoring with real numbers and compare against the proposal-only
      results from run-20260808-0559 to see whether the A3-vs-A1 conclusions hold
- [ ] Full convergence training (multi-day, much larger GPU-hour cost) is explicitly out of
      scope for this phase — noted as a possible future Phase if warranted by pilot results

## Phase 3: Adapter implementation

**Status:** future

- [ ] Implement HF/DeepSpeed/Megatron adapters as config generation + validation tools

## Phase 4: Agent orchestration

**Status:** future

- [ ] Implement orchestrator and experiment-manager agents

## Phase 5: Hardening & release

**Status:** future

- [ ] CI
- [ ] Test coverage
- [ ] Packaging
- [ ] Docs polish
- [ ] v1.0 release
