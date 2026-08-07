# Contributing

Thanks for your interest in MoE Agentic Development. This project ships agentic
skills for MoE engineering plus the tooling to load, deploy, and benchmark them.
Below is how to contribute a skill, keep it compliant, and get it merged.

## How to contribute a skill

1. Create `skills/<name>/SKILL.md` with valid YAML frontmatter — see
   [docs/skill-spec.md](docs/skill-spec.md) for the exact schema.
2. Add the optional subdirectories per the spec: `knowledge/` (deep,
   concrete reference material), `tools/` (runnable scripts), and `examples/`
   (complete, worked examples).
3. Run `moe-skills validate` and make sure everything passes:
   ```
   moe-skills validate
   ```
4. Verify the skill is discoverable and inspectable:
   ```
   moe-skills list
   moe-skills info <name>
   ```

## Skill spec compliance

- The skill must pass `moe-skills validate`.
- Scripts under `tools/` must be stdlib-only (no third-party dependencies).
- Markdown must be well-formed.
- No files outside the skill's own directory.

## Tests

Run the test suite with `pytest` (the `tests/` directory exists):

```
pytest
```

Keep the tests green when you open a pull request.

## Pull request process

- Branch from `main`.
- Use conventional commits.
- One skill per PR.
- Review checklist: frontmatter valid, `moe-skills validate` passes, tests
  green, no files touched outside the skill directory, examples consistent.

## Code of conduct

All contributors are expected to treat others with respect; see the project
maintainers for concerns.
