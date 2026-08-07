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
- [x] Agent role docs (orchestrator, experiment-manager) — placeholders
- [x] Adapters (HF/DeepSpeed/Megatron) — placeholders
- [ ] Running the first benchmark wave
- [ ] Integrating feedback into skills

## Phase 2: Benchmark execution

**Status:** future

- [ ] Run controlled benchmark baseline vs treatment
- [ ] Statistical analysis
- [ ] Publish results in benchmarks/results

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
