# Benchmark

This document defines the research design for measuring whether the MoE skill
suite improves agent performance on MoE engineering tasks. All experiments are
controlled agent experiments: the same model, the same hardware, and the same
task set, run under different *context conditions*, with outputs scored against
pre-registered rubrics.

## The problem this study is actually about

The naive version of this question — "does an agent with MoE skills beat an
agent with no MoE skills?" — is not worth asking. A modern agent has web search,
documentation retrieval, and a large amount of MoE literature in its weights. It
can look up the Switch Transformer capacity factor, the DeepSeek-MoE
shared-expert design, or the Megatron expert-parallel flags on demand. Any study
that withholds search from the baseline and then credits the difference to
skills is measuring *access to information*, not the value of the skills, and
will overstate the effect.

So the study is designed around a stronger claim:

> **Research question:** Given an agent that can already retrieve MoE knowledge
> on demand, does packaging that domain as procedural skills produce measurably
> better, cheaper, and more reproducible MoE engineering work than retrieval
> alone?

This reframes the contribution. Skills are not a substitute for information
access; they are a substitute for *rediscovering a procedure from scratch on
every run*. The five mechanisms below are the specific ways that matters, and
each is stated as a falsifiable hypothesis with a metric attached.

### What search does not solve

**1. Declarative vs procedural knowledge.** Search returns statements of fact —
papers, docs, blog posts. MoE engineering failures are mostly procedural: the
order in which you triage a NaN loss, which knob to hold fixed while moving
another, what to check before concluding "expert collapse". A retrieved paper
tells you what a capacity factor is; it does not tell you that when tokens are
being dropped you adjust capacity factor *before* touching the aux-loss
coefficient, because the latter changes the routing distribution you are trying
to measure. Skills encode ordered decision procedures with defaults; retrieval
does not. (**H1**)

**2. Retrieval is non-deterministic, so retrieval-grounded work is not
reproducible.** The same prompt on two days returns different sources, and
paraphrasing the query changes the ranking. An agent that derives its capacity
factor from whatever happened to rank first produces a different architecture
each run. A skill is a fixed artifact under version control: the same input
yields the same procedure. The prediction is not only a higher mean but a
**lower run-to-run variance**, which for an experiment pipeline matters more
than the mean. (**H2**)

**3. Retrieval costs tokens and wall-clock.** Rediscovering the same domain
procedure on every run burns search calls, context window, and time. A skill
loads once. The prediction is fewer tool calls, fewer tokens, and less
wall-clock to a first valid artifact, at equal or better quality. (**H3**)

**4. Sources conflict, and search does not arbitrate.** MoE literature
genuinely disagrees: capacity factors from 1.0 to 2.0, aux-loss coefficients
across two orders of magnitude, top-1 vs top-2 routing, whether to use a shared
expert. Search surfaces all of it with no basis for choosing, and agents
resolve the conflict by picking whichever source is most recent or most
prominent — often mixing incompatible recommendations from different papers
into a single incoherent config. A skill commits to a defended default and
states the conditions under which to deviate. The prediction is fewer
**internally inconsistent** configurations. (**H4**)

**5. Search returns prose; skills ship executable verification.** The skill
directories carry tools and evaluators — parameter-math checkers, config
validators, deterministic scorers. No amount of retrieval gives the agent a
program that checks whether its own parameter arithmetic closes. The prediction
is a lower numeric-error rate in produced artifacts. (**H5**)

There is also a deployment argument that the benchmark does not measure but that
motivates the work: training clusters are frequently egress-restricted or
air-gapped, so the environment where the MoE work actually happens is often one
where search is unavailable. A versioned skill directory ships with the repo.

### What search *does* solve, and what that means for the claim

Search genuinely covers recency (a technique published after the model's
cutoff), breadth (an unusual framework flag), and specific API syntax. The skill
suite is not expected to beat retrieval on those, and the study should not claim
it does. Where the arms tie, we report a tie. The honest form of a positive
result here is: *equal or slightly better quality, materially lower variance and
cost* — and if the treatment arm does not beat the search-enabled baseline on
quality, that is a publishable finding about the limits of skill packaging, not
a failure to be tuned away.

## Design: four arms

The central methodological change is that **the baseline is search-enabled**.
The no-search arm is retained only as a floor, to quantify how much of any
observed gain is plain information access.

| Arm | Condition | Web search / doc retrieval | MoE skills | Role |
|-----|-----------|---------------------------|-----------|------|
| **A0** | Bare model | No | No | Floor. Measures the task's intrinsic difficulty for the model alone. |
| **A1** | Retrieval baseline | Yes | No | **The real control.** A competent engineer-agent that looks things up. |
| **A2** | Skills, offline | No | Yes | Isolates the skills' contribution with retrieval held out; models the air-gapped cluster. |
| **A3** | Skills + retrieval | Yes | Yes | The realistic deployed configuration; tests whether skills and search compose or interfere. |

**Primary comparison: A3 vs A1.** This is the claim that matters — the deployed
configuration against a strong, search-enabled baseline. It is pre-registered as
the primary endpoint; all others are secondary.

Secondary comparisons and what each isolates:

- **A1 − A0** — the *retrieval-only gain*. Reporting this is what keeps the
  study honest: it states plainly how much of the total effect is available from
  search alone, and it is the number a skeptical reader will look for first.
- **A2 − A0** — the skills' contribution absent any retrieval.
- **A2 vs A1** — skills versus search as substitutes for the same knowledge.
- **A3 − A2** — what retrieval still adds on top of skills (expected to be
  positive on recency-dependent items and near zero elsewhere).
- **(A3 − A1) − (A2 − A0)** — an interaction estimate: whether skills and search
  are complementary, redundant, or actively conflicting.

Everything else is held constant across arms: same model checkpoint and
version, same hardware and software stack, same task prompts, same seeds, same
scoring. Runs are paired by (task, seed) across all four arms.

### Controlling the search confound

An arm-by-arm difference in *tool access* is a confound unless it is managed
explicitly:

- **Equal tool budget.** A1 and A3 get the same cap on search calls and the same
  retrieval backend; A0 and A2 have retrieval disabled at the harness level, not
  by instruction in the prompt. Instructing a model not to search is not the
  same as removing the tool, and produces contaminated arms.
- **Log every query.** Every search query, the URLs returned, and the retrieved
  snippets are recorded per run. Retrieval is a moving target, so a run without
  its query log is not reproducible and is excluded.
- **Pin what can be pinned.** Where the retrieval backend supports it, snapshot
  the index or cache responses so the A1/A3 arms can be replayed. If it cannot,
  record the wall-clock date of each run and treat cross-date comparisons as a
  stated limitation.
- **Do not let skills smuggle in search.** Skill instructions must not tell the
  agent to search; if a skill's text drives retrieval behavior, A2 and A3 stop
  being separable.
- **Prompt parity.** The task prompt is byte-identical across arms. The only
  differences are the loaded skill directories and whether the retrieval tool is
  registered.

## Hypotheses

Pre-registered, with the arm comparison and metric for each. H1–H5 map to the
five mechanisms above.

| ID | Hypothesis | Comparison | Primary metric |
|----|-----------|-----------|----------------|
| **H1** | Skills improve *procedural* task quality beyond retrieval. | A3 > A1 | Task 3 diagnosis accuracy; Task 1 rubric total |
| **H2** | Skills reduce run-to-run variance. | Var(A3) < Var(A1) | SD of per-task score across seeds |
| **H3** | Skills reduce cost at equal-or-better quality. | A3 < A1 | Tool calls, total tokens, wall-clock to first valid artifact |
| **H4** | Skills reduce internally inconsistent configurations. | A3 < A1 | Inconsistency rate (see below) |
| **H5** | Skills reduce numeric errors in artifacts. | A3 < A1 | Numeric-consistency error rate |
| **H0** | No difference on any endpoint. | — | — |

H2 and H3 are the hypotheses most likely to survive a strong baseline, and the
design is powered with that in mind: variance and cost endpoints are measured on
every run, so they carry the full sample.

## Tasks

The task set is chosen so that retrieval alone is *not* sufficient — each task
requires committing to a coherent set of interacting choices, which is where
declarative lookup underperforms. Tasks 5 and 6 are added specifically to
discriminate skills from search.

| Task | Prompt | Metrics |
|------|--------|---------|
| Task 1: Architecture Design | "Design a 1B dense-equivalent MoE model" | correctness (param math), efficiency, expert-utilization awareness |
| Task 2: Training Setup | Create a training config consistent with the architecture | successful launch, memory efficiency, throughput |
| Task 3: Debugging | Diagnose broken runs (expert collapse, NaN, OOM, comm bottleneck) | diagnosis accuracy, fix success, time-to-diagnosis |
| Task 4: Optimization | Improve throughput of a working run | before/after tokens/sec, GPU util, memory |
| Task 5: Conflicting-guidance resolution | Choose capacity factor, aux-loss coefficient, and top-k for a stated hardware budget, and justify the choice against the disagreement in the literature | internal consistency, justification quality, deviation from stated conditions |
| Task 6: Constrained-hardware design | Design under an unusual constraint (e.g. limited interconnect bandwidth, fixed expert-parallel degree) that no published recipe matches directly | correctness, constraint satisfaction, param math |

Tasks 5 and 6 exist because Tasks 1–4 are partly addressable by retrieval —
a well-searched agent can assemble a plausible answer from published recipes.
Task 5 forces arbitration between conflicting sources; Task 6 forces
extrapolation beyond any single published recipe. If skills beat search
anywhere, these are where the gap should be largest, and if the gap is absent
here it is unlikely to be real elsewhere.

## Metrics

### Quality (per task)

Owned by the `moe-benchmarking` skill, which holds the rubrics and evaluators:

- Rubric checklists — `skills/moe-benchmarking/SKILL.md` (Task library and
  "Metrics definitions and scoring rubrics").
- Evaluator scripts — `skills/moe-benchmarking/evaluators/score_architecture.py`
  and `score_debugging.py` score outputs deterministically, so scoring cannot be
  influenced by knowing which arm produced an output.

### Cost (every run, every arm)

- **Tool calls** — total, and search calls specifically.
- **Total tokens** — prompt + completion, including retrieved content pulled
  into context.
- **Wall-clock to first valid artifact** — time until an output that passes the
  task's validity check (not the rubric score).

### Reliability (across seeds within an arm)

- **Score SD** per task per arm — the H2 endpoint.
- **Numeric-consistency error rate** — fraction of artifacts where the stated
  numbers do not close under the task's own formulas (e.g. total parameters
  inconsistent with expert count × expert size, memory estimate inconsistent
  with the stated batch geometry). Checked mechanically.
- **Internal-inconsistency rate** — fraction of artifacts combining mutually
  incompatible recommendations (e.g. a top-1 routing choice paired with a
  capacity factor justified for top-2). Checked against a pre-registered list of
  incompatible pairs, so the check stays deterministic.

## Statistical design

- **Sample size:** n ≥ 5 paired runs per arm per task, more with budget. With
  four arms and six tasks this is ≥ 120 runs at n = 5.
- **Design:** paired across arms by (task, seed); the unit of analysis is the
  per-pair difference.
- **Primary endpoint:** A3 vs A1 on task quality, one pre-registered test.
- **Multiplicity:** all other comparisons are secondary and reported with an
  explicit correction (Holm) over the pre-registered secondary family. Secondary
  results are not presented as confirmatory.
- **Effect size:** Cohen's d on the paired differences.
- **Variance endpoint:** SD ratio between arms with bootstrap CIs (a variance
  ratio test on scores that are not normally distributed would be misleading).
- **Reporting:** mean ± 95% CI per metric per arm, plus the paired difference
  and its CI.
- **Pre-registration:** hypotheses, rubrics, thresholds, the incompatible-pair
  list, and the exclusion rules are fixed before any run is scored.

## Results table template

| Task | Metric | A0 bare | A1 search | A2 skills | A3 skills+search | A3−A1 | Cohen's d |
|------|--------|---------|-----------|-----------|------------------|-------|-----------|
| task1 | correctness (total /25) | | | | | | |
| task2 | successful launch | | | | | | |
| task3 | accuracy (0-1) | | | | | | |
| task4 | throughput (tok/s) | | | | | | |
| task5 | internal consistency | | | | | | |
| task6 | constraint satisfaction | | | | | | |
| all | score SD (H2) | | | | | | |
| all | total tokens (H3) | | | | | | |
| all | tool calls (H3) | | | | | | |
| all | numeric-error rate (H5) | | | | | | |

## Threats to validity

- **Contamination.** Task prompts or reference solutions may appear in the
  model's training data or be directly retrievable. Rotate the task set and
  prefer constructed scenarios (Tasks 5 and 6) over published ones.
- **Retrieval drift.** A1/A3 results depend on what the index returns on the day
  of the run. Mitigated by query logging and caching; stated as a limitation
  where caching is impossible.
- **Ceiling effects on Tasks 1–2.** A search-enabled baseline may already score
  near the rubric maximum, leaving no room to detect a difference. Verify
  headroom before the full wave; if A1 saturates, the finding is that skills add
  nothing *there*, and it is reported as such.
- **Skill-rubric circularity.** The rubrics and the skills were written in the
  same repository, so rubrics may unconsciously encode what the skills happen to
  do. Rubrics must be justified against external MoE literature, and any
  criterion that only the skills' phrasing satisfies is removed before scoring.
  This is the most serious threat to the study and is checked by a reviewer who
  did not author the skills.
- **Cost metrics are runtime-dependent.** Token and wall-clock figures are
  specific to the harness and model version; report the environment and treat
  cross-version comparisons as invalid.

## Running it

The run protocol, scoring, and analysis are executed by the `moe-benchmarking`
skill (see `skills/moe-benchmarking/SKILL.md` and
`skills/moe-benchmarking/examples/benchmark-protocol.md`). Raw outputs and query
logs land in `benchmarks/baseline/` (arms A0 and A1) and
`benchmarks/with-skills/` (arms A2 and A3), one subdirectory per arm; aggregated
results and analysis go in `benchmarks/results/`.
