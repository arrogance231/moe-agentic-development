# MoE Agentic Development

Agentic skills for Mixture-of-Experts engineering.

GPU compute for all training and benchmark runs is provided by the
**[AMD AI Developer Program](https://developer.amd.com/)** — see
[Acknowledgments](#acknowledgments) and [CONTRIBUTORS.md](CONTRIBUTORS.md).

## What it is

An agentic skill ecosystem for MoE engineering: five domain skills (architecture
design, training setup, debugging, performance optimization, benchmarking) that
can be loaded into Claude Code, OpenCode, or custom agents, backed by a Python
skill-loader/deployer CLI (`moe-skills`). Each skill packages its instructions
as a `SKILL.md` with YAML frontmatter plus subdirectories such as
`knowledge/`, `tools/`, `examples/`, and `evaluators/` as appropriate, so any
agent runtime that reads frontmatter-driven skill directories can use them.

## Research question

A modern agent already has web search and a large amount of MoE literature in
its weights. Beating an agent that has *no* access to MoE knowledge would only
measure access to information, so that is not the question this repository asks.
The baseline is search-enabled:

> Given an agent that can already retrieve MoE knowledge on demand, does
> packaging that domain as procedural skills produce measurably better,
> cheaper, and more reproducible MoE engineering work than retrieval alone?

The claim is that retrieval returns *declarative* knowledge — statements of fact,
often conflicting — while the hard part of MoE engineering is *procedural*:
which knob to move first, what to hold fixed, which default to commit to when
the literature disagrees. Search also re-pays its cost on every run, returns
different sources on different days, and ships no executable verification. The
benchmark tests each of those as a separate hypothesis (H1–H5) against a
search-enabled control.

The repository ships a benchmark harness with a four-arm design — bare model,
search-only, skills-only, and skills+search — see [BENCHMARK.md](BENCHMARK.md)
for the design, hypotheses, and threats to validity.

## Architecture

```
Claude Code · OpenCode · Custom agent
        │               │            │
        │   read SKILL.md frontmatter, invoke skill resources
        ▼               ▼            ▼
┌───────────────────────────────────────────────┐
│             moe-skills CLI / loader           │
│   list · validate · info · deploy             │
└───────────────────┬───────────────────────────┘
                    │ deploy (claude/opencode/agents)
                    ▼
┌───────────────────────────────────────────────┐
│                   skills/                     │
│  moe-architecture   moe-debugging             │
│  moe-training       moe-performance           │
│  moe-benchmarking                             │
│  each: SKILL.md + subdirs (tools, evaluators…)│
└───────────────────┬───────────────────────────┘
                    │ tasks
                    ▼
┌───────────────────────────────────────────────┐
│              Benchmark harness                │
│  tasks → agent runs → evaluators → results    │
│  (baseline/ vs with-skills/ → results/)       │
└───────────────────────────────────────────────┘
```

## Quickstart

1. **Install** the package and CLI:
   ```
   pip install -e .
   ```
   (or `uv pip install -e .`)

2. **List the available skills:**
   ```
   moe-skills list
   ```

3. **Deploy skills to an agent runtime** — see the deploy help, then run an
   example:
   ```
   moe-skills deploy --help
   moe-skills deploy --target all --dry-run
   moe-skills deploy --target claude
   ```

4. **Validate the skill suite:**
   ```
   moe-skills validate
   ```

5. **Use the skills.** Deployment copies each skill into the runtime's skills
   directory: `.claude/skills` for Claude Code, `.opencode/skills` for
   OpenCode, and `.agents/skills` for generic custom agents. Any agent runtime
   that loads skill directories containing `SKILL.md` frontmatter can then pick
   up the skills. You can inspect an individual skill's frontmatter and
   instructions with:
   ```
   moe-skills info moe-architecture
   ```

## Skills

| Skill | Purpose | Phase |
|-------|---------|-------|
| `moe-architecture` | Design MoE architectures (expert counts, routing, capacity) | Phase 1 |
| `moe-training` | Generate launchable training configurations | Phase 1 |
| `moe-debugging` | Diagnose training failures (expert collapse, NaN, OOM) | Phase 1 |
| `moe-performance` | Optimize training throughput and efficiency | Phase 1 |
| `moe-benchmarking` | Run controlled agent benchmark experiments | Phase 1 |

## Benchmarks

The benchmark harness measures whether the MoE skills improve agent performance
over a **search-enabled** baseline, using a four-arm design (bare / search /
skills / skills+search) paired by task and seed. It scores quality, cost (tokens,
tool calls, wall-clock), and reliability (run-to-run variance, numeric and
internal-consistency error rates). See [BENCHMARK.md](BENCHMARK.md) for the
design, hypotheses, tasks, metrics, and threats to validity.

## Roadmap

Current phase: **Phase 1 — Skill framework**.

| Phase | Scope | Status |
|-------|-------|--------|
| **0. Research** | Literature review, skill-domain selection, SKILL.md format validation | Complete |
| **1. Skill framework** | Loader + `moe-skills` CLI, five skills, skill spec, benchmark harness, four-arm research design | **Active** |
| **2. Benchmark execution** | Headroom check, all four arms, statistical analysis, published results | Future |
| **3. Adapters** | HF / DeepSpeed / Megatron config generation and validation | Future |
| **4. Agent orchestration** | Orchestrator and experiment-manager agents | Future |
| **5. Hardening & release** | CI, coverage, packaging, docs polish, v1.0 | Future |

Phase 1 is complete except for the run-harness work the four-arm design
requires: Tasks 5 and 6, cost instrumentation (tool calls, tokens, wall-clock),
search-query logging and retrieval caching for the search-enabled arms, and an
independent review of the rubrics for skill-rubric circularity.

See [ROADMAP.md](ROADMAP.md) for the per-item checklists.

## Documentation

- [docs/README.md](docs/README.md) — documentation index
- [docs/skill-spec.md](docs/skill-spec.md) — the SKILL.md specification
- [docs/architecture.md](docs/architecture.md) — framework architecture
- [ROADMAP.md](ROADMAP.md) — project roadmap
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to contribute
- [CONTRIBUTORS.md](CONTRIBUTORS.md) — people and organizations behind the project

## Acknowledgments

**[AMD AI Developer Program](https://developer.amd.com/)** — main contributor,
for providing free access to AMD GPU compute. The benchmark design calls for a
four-arm, six-task, n≥5 experiment — on the order of 120 agent runs plus the
training and profiling work behind Tasks 2 and 4 — and none of it could be
tested or trained without that hardware. Every empirical result published here
runs on AMD GPUs through the program.

This is an acknowledgment of compute support. The AMD AI Developer Program does
not author, review, or endorse this work.

## License

[Apache-2.0](LICENSE).
