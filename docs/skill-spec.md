# Skill Specification

This document is the formal specification for authoring skills in this
repository. Every skill is a directory under `skills/<name>/` containing a
`SKILL.md` file with YAML frontmatter, validated by the skill loader in
`src/moe_agentic/skill_loader.py` and the `moe-skills validate` CLI command.

## SKILL.md format

A skill is a Markdown file with a YAML frontmatter block at the very top. The
frontmatter sits between two `---` fences on their own lines; the rest of the
file is the skill's body markdown.

```
---
name: example-skill
description: A short description of what the skill does.
---

# example-skill

Instructions for the agent go here.
```

## Frontmatter fields

| Field | Required? | Type | Description | Validation |
|-------|-----------|------|-------------|------------|
| `name` | required | string | Kebab-case skill identifier | matches `^[a-z0-9]+(-[a-z0-9]+)*$`; ≤ 64 characters |
| `description` | required | string | 2-3 sentences describing the skill's purpose | must be present |
| `license` | optional | string | SPDX license identifier | valid SPDX expression (e.g. `Apache-2.0`) |
| `compatibility` | optional | list of strings | Runtime ids the skill supports | entries from: `claude-code`, `opencode`, `generic` |
| `metadata` | optional | map of strings | Arbitrary key-value metadata (e.g. domain, phase, version) | map of string keys/values |
| `argument-hint` | optional | string | Hint shown when the skill is invoked | string |

## Directory conventions

Each skill lives at `skills/<name>/` with a `SKILL.md` plus optional
subdirectories:

| Dir | Purpose | Contents |
|-----|---------|----------|
| `skills/<name>/` | The skill root | `SKILL.md` with frontmatter and instructions |
| `knowledge/` | Deep, concrete reference material | domain notes, references, lookups |
| `tools/` | Runnable scripts the agent can invoke | stdlib-only Python or shell scripts |
| `examples/` | Complete, worked examples | input + output pairs demonstrating the skill |

## Validation rules

`moe-skills validate` checks each skill directory:

- [ ] Name matches `^[a-z0-9]+(-[a-z0-9]+)*$` and is ≤ 64 characters.
- [ ] Frontmatter parses as YAML between `---` fences.
- [ ] `description` is present.
- [ ] Directories match the conventions above (no unexpected structure).
- [ ] Scripts in `tools/` have no non-stdlib dependencies.
- [ ] Markdown is well-formed.

## Quality bar

Every skill should provide:

- **Your role** — what the agent becomes when it uses the skill.
- **When to use** — the situations and request types the skill covers.
- **Required context** — what the agent needs before starting, and what to
  elicit if missing.
- **Inputs** — the inputs the skill consumes.
- **Workflow** — the steps the agent follows.
- **Expected output** — the artifacts the skill produces.
- **Evaluation criteria** — how the output quality is judged.

In addition:

- Knowledge files are deep and concrete (real numbers, real failure modes).
- Tools are runnable, stdlib-only scripts.
- Examples are complete and internally consistent (outputs match the described
  inputs and workflow).
