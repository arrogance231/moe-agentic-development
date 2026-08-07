# MoE Agentic Development

Agentic skills for Mixture-of-Experts engineering.

## What it is

An agentic skill ecosystem for MoE engineering: five domain skills (architecture
design, training setup, debugging, performance optimization, benchmarking) that
can be loaded into Claude Code, OpenCode, or custom agents, backed by a Python
skill-loader/deployer CLI (`moe-skills`). Each skill packages its instructions
as a `SKILL.md` with YAML frontmatter plus subdirectories such as
`knowledge/`, `tools/`, `examples/`, and `evaluators/` as appropriate, so any
agent runtime that reads frontmatter-driven skill directories can use them.

## Research question

> Do domain-specific skills measurably improve MoE engineering performance vs a
> general LLM?

The repository ships a benchmark harness designed to answer this question with
controlled, paired agent experiments — see [BENCHMARK.md](BENCHMARK.md).

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

The benchmark harness measures whether the MoE skills measurably improve agent
performance on MoE engineering tasks, using a paired baseline-vs-treatment
design. See [BENCHMARK.md](BENCHMARK.md) for the methodology, tasks, and
metrics.

## Documentation

- [docs/README.md](docs/README.md) — documentation index
- [docs/skill-spec.md](docs/skill-spec.md) — the SKILL.md specification
- [docs/architecture.md](docs/architecture.md) — framework architecture
- [ROADMAP.md](ROADMAP.md) — project roadmap
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to contribute

## License

[Apache-2.0](LICENSE).
