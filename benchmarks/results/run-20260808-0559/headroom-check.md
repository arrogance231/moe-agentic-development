# Headroom check — arm A1, tasks 1-2, n=2

Per skills/moe-benchmarking/examples/benchmark-protocol.md: run n=2 of arm
A1 only on tasks 1-2 before committing to the full n=5x4-arm wave, to catch
ceiling effects early.

## Critical finding during this check (fixed before results below)

The first attempt at task2/seed2 (harness commit 964cf5f config) timed out
after 300s having burned 44 tool calls / 524,194 tokens. Inspecting its
partial output showed the model using filesystem tools to read
`skills/moe-benchmarking/evaluators/score_architecture.py`, the (nonexistent
at the time) task2 scorer, skill worked examples, and even the prior A1
task2 run's own output file — explicitly reasoning "Let me check the skill
rubrics to produce a coherent, high-scoring config." This is the
skill-rubric-circularity threat that methodology.md flags as the most
serious validity risk, occurring through an unintended channel: the arm
configs disabled `bash`/`edit`/`write`/`webfetch` but left OpenCode's
built-in `read`/`grep`/`glob`/`list` tools enabled, so the agent could
browse the harness's own repo, including scoring code, from inside the run.

**Fix applied**: all four arm configs
(`benchmarks/harness/configs/{A0,A1,A2,A3}/opencode.jsonc`) now also disable
`read`, `grep`, `glob`, `list`. Verified with a probe run
("list the files in the current directory using any tool available") that
produced zero tool-call events post-fix. All headroom-check runs below are
post-fix, clean re-runs (the pre-fix task1/task2 outputs were discarded and
re-generated). **This same check must be re-verified before the full wave**
and is the top item to watch for in run-log.md during the real run — if any
run shows filesystem tool calls, treat it as a contamination failure per
the protocol, not just a cost outlier.

## task1 (Architecture Design) — score_architecture.py, /25, PASS >= 15

| seed | total | passed |
|------|-------|--------|
| 1    | 21    | true   |
| 2    | 25    | true   |

Both runs pass comfortably, and seed 2 hit the exact rubric ceiling (25/25).

## task2 (Training Setup) — schema validity + consistency (thresholds.md)

| seed | JSON parses | required keys present | wall_clock | tokens_total |
|------|-------------|------------------------|-----------|--------------|
| 1    | yes         | yes (0 missing)        | 51.8s     | 13,973       |
| 2    | yes         | yes (0 missing)        | 31.4s     | 6,999        |

Both runs produce schema-valid, complete configs on the first attempt with
plausible-looking numbers (param math closes, top_k <= num_experts in both).

## Verdict: ceiling-effect risk on task1 — CONFIRMED, action needed

task1 scored 21/25 and 25/25 out of only n=2 A1-only samples — already at or
within 4 points of the rubric maximum. Per BENCHMARK.md's/methodology.md's
own criterion ("if A1 already scores at or near the rubric maximum, the
task has no headroom to detect a difference... verify headroom before the
full wave"), **task1 as currently scored by score_architecture.py has
materially limited headroom for detecting an A3-vs-A1 skill effect** with
this model (DeepSeek V4 Flash) — there isn't much room left for skills to
push scores higher, and a ceiling near 25 will compress any true effect
size and bias Cohen's d toward zero on this task specifically.

task2 shows no comparable ceiling signal (it's scored 0/1 on
schema/consistency, not a 0-25 rubric), but with n=2 both runs already
succeed outright — so task2's "successful launch" endpoint may also have
limited headroom to distinguish arms, though this is a binary metric so the
concern is different (near-100% success rate under this proxy metric
regardless of arm, not a graded-rubric ceiling).

**Recommendation, applied going into the full wave**: proceed with the full
wave as designed (per BENCHMARK.md, tasks 5/6 exist specifically to
discriminate skills from retrieval where 1-4 are more saturable), but flag
task1 and task2 in summary.md as *lower-power* endpoints a priori due to
this ceiling signal, and weight interpretation of the A3-vs-A1 primary
endpoint accordingly if it is computed on task1/task2 specifically. This
is the documented, pre-registered call per methodology.md's own guidance —
not a decision to drop or reweight tasks post hoc after seeing full-wave
data.
