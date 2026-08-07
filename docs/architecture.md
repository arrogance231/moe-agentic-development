# Architecture

The MoE Agentic Development framework turns a general LLM into a competent MoE
engineering assistant: five domain skills, a skill loader and deployer exposed
through the `moe-skills` CLI, and a benchmark harness that measures whether the
skills measurably improve MoE engineering.

## Components

1. **Skill loader** — `src/moe_agentic/skill_loader.py`. Discovers, validates,
   parses, and loads `SKILL.md` files: YAML frontmatter parsing, name
   validation, and detection of the optional `knowledge/`, `tools/`,
   `examples/` subdirectories.
2. **Deployer** — `src/moe_agentic/deploy.py`. Copies skills into target
   runtimes, project-local or user-global, for `claude-code` (`.claude/skills`),
   `opencode` (`.opencode/skills`), and generic (`.agents/skills`) agents.
3. **CLI** — `src/moe_agentic/cli.py`. The `moe-skills` command with four
   subcommands: `list`, `validate`, `info`, and `deploy`.
4. **Skill content** — `skills/`. Five skills, each with `SKILL.md` plus
   subdirectories such as `knowledge/`, `tools/`, `examples/`, and
   `evaluators/` as appropriate.
5. **Agents** — `agents/`. Orchestrator and experiment-manager roles,
   Phase 1 placeholders.
6. **Adapters** — `adapters/`. HF/DeepSpeed/Megatron config adapters,
   Phase 1 placeholders.
7. **Benchmarks** — `benchmarks/`. Evaluator scripts plus the `baseline/`,
   `with-skills/`, and `results/` output directories.

## ASCII diagram

```
                  Authoring                     Runtime
┌────────────────────────────────┐      ┌──────────────────────────────┐
│  skills/<name>/                │      │  Agents: Claude Code,        │
│    SKILL.md  (frontmatter)     │      │  OpenCode, custom             │
│  knowledge/ tools/ evaluators/…│      │  (read SKILL.md frontmatter,  │
└───────────────┬────────────────┘      │   invoke tools/knowledge)     │
                │                       └──────────────┬───────────────┘
                │                                      │
                ▼                                      │ loads
┌────────────────────────────────┐                     │
│  moe-skills CLI / loader       │                     │
│  list · validate · info        │                     │
└───────────────┬────────────────┘                     │
                │ deploy (claude/opencode/agents)      │
                ▼                                      │
┌────────────────────────────────┐                     │
│  Runtime skills dirs           │                     │
│  .claude/skills  .opencode/    │◄────────────────────┘
│  skills  .agents/skills        │
└───────────────┬────────────────┘
                │ execute tasks
                ▼
┌────────────────────────────────┐
│  Benchmark harness             │
│  tasks → agent runs            │
│  (baseline/ vs with-skills/) → │
│  evaluators → results/         │
└────────────────────────────────┘
```

## Data flow

**How a skill is authored and shipped:**

1. Author the skill directory under `skills/<name>/`.
2. Validate it: `moe-skills validate` (plus `moe-skills list` /
   `moe-skills info` to inspect).
3. Deploy it: `moe-skills deploy --target <claude|opencode|agents|all>` copies
   the skill into the runtime's skills directory.

**How an agent uses a skill:**

1. The runtime discovers the skill directory.
2. It reads the `SKILL.md` frontmatter (name, description, compatibility,
   metadata) and instructions.
3. It invokes the skill's scripts (under `tools/`, `evaluators/`, or similar)
   and consults `knowledge/` and `examples/` as directed.

## Phase 1 scope

**Implemented now:** the skill framework — loader, deployer, CLI, the five
authored skills, the skill-spec documentation, and the benchmark harness
(evaluators + methodology).

**Placeholders:** `agents/` (orchestrator, experiment-manager) and `adapters/`
(HF/DeepSpeed/Megatron) are documented roles, not yet implemented.

**Future:** benchmark execution waves (Phase 2), adapter implementation
(Phase 3), agent orchestration (Phase 4), and hardening & release (Phase 5) —
see [ROADMAP.md](../ROADMAP.md).
