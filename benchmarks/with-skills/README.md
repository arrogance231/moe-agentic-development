# benchmarks/with-skills

Outputs for the two skills-loaded arms, one subdirectory per arm. Same agent,
same tasks, same seeds as `benchmarks/baseline/`:

- `A2/` — skills loaded, retrieval disabled. Isolates the skills' contribution
  and models the egress-restricted training cluster.
- `A3/` — skills loaded, retrieval enabled. The deployed configuration and the
  primary comparison against arm A1.

Layout matches `benchmarks/baseline/`; A3 runs carry query logs
(`taskN_runM.queries.jsonl`) on the same terms as A1.

Populated by the `moe-benchmarking` skill.
