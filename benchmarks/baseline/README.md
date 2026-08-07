# benchmarks/baseline

Baseline condition outputs. Each run: an agent WITHOUT MoE skills executing a
task prompt. Layout: `taskN_runM.md` + scored metrics.

Populated by the `moe-benchmarking` skill: each run saves the raw agent output
here and the per-run scored metrics alongside it, named `taskN_runM.metrics.json`.
