# Research Methodology for LLM-as-Engineer Evaluation

This document is the methodological foundation for the `moe-benchmarking`
skill. It explains why the comparisons are designed the way they are and how to
report results honestly.

## Choosing the right baseline

Before anything else: **the baseline must be allowed to search.**

The tempting comparison is "agent with MoE skills" vs "agent with nothing". It
is also close to meaningless. A modern agent has web search, documentation
retrieval, and substantial MoE literature in its weights; it can look up the
Switch Transformer capacity factor or the Megatron expert-parallel flags on
demand. A study that withholds that and credits the resulting gap to skills has
measured information access and mislabelled it.

The defensible claim is narrower and more interesting: *given* an agent that can
already retrieve the knowledge, does packaging the domain as procedural skills
still help? The mechanisms that could make it help — and they are what the
metrics must target — are:

1. **Procedural vs declarative.** Retrieval returns facts. MoE failures are
   procedural: triage order, what to hold fixed, which knob moves first. A paper
   defines capacity factor; it does not tell you to adjust it before the
   aux-loss coefficient because the latter perturbs the routing distribution you
   are measuring.
2. **Determinism.** Search results drift across days and query phrasings, so
   retrieval-grounded work is not reproducible. A versioned skill is. Predict
   *lower variance*, not just a higher mean.
3. **Cost.** Rediscovering the same procedure every run costs tool calls,
   context, and wall-clock. Measure all three.
4. **Arbitration.** The MoE literature genuinely disagrees (capacity factors
   1.0-2.0, aux-loss coefficients across two orders of magnitude, top-1 vs
   top-2). Search surfaces the disagreement without resolving it, and agents
   often blend incompatible recipes into one incoherent config. A skill commits
   to a defended default and names the conditions for deviating.
5. **Executable verification.** Skills ship evaluators and checkers; retrieved
   prose cannot verify that an agent's own parameter arithmetic closes.

Search legitimately wins on recency, breadth, and exact API syntax. Do not claim
otherwise. Where the arms tie, report a tie — equal quality at lower variance
and lower cost is a real result and does not need inflating.

## Controlled comparison

The four arms (A0 bare, A1 search, A2 skills, A3 skills+search) must differ in
**nothing but retrieval access and skills loaded**:

- **Same model** — the exact same LLM checkpoint and version.
- **Same hardware** — the same GPUs, node topology, and software stack.
- **Same tasks** — the same prompts, inputs, and expected outputs.
- **Same seeds** — the same random seeds where the runtime allows them.

The treatment arms add the MoE skill suite (the skills under `skills/`) on top
of the same agent; everything else is held constant. If any other dimension
differs between arms, the experiment is confounded and its conclusions are
unsupportable.

Retrieval access is the one dimension that varies *by design*, so it needs its
own controls:

- **Disable search at the harness level** for A0/A2 — unregister the tool. Do
  not tell the model not to search: it may search anyway, and being told not to
  changes its behavior in other ways.
- **Equal budget.** A1 and A3 share a retrieval backend and a per-run tool-call
  cap.
- **Log everything.** Queries, returned URLs, retrieved snippets. Retrieval is a
  moving target; a run without its query log cannot be replayed and is excluded.
- **Cache or snapshot** responses where the backend permits replay; otherwise
  record run dates and declare cross-date comparison a limitation.
- **Keep skills search-silent.** No skill may instruct the agent to search, or
  the skills-only and skills+search arms stop being separable.
- **Always report A1 - A0**, the retrieval-only gain. It tells the reader how
  much of the total effect plain search already buys, and its absence is the
  first thing a skeptical reviewer will notice.

## Blinding and pre-registration

Human scoring is a source of bias. To keep scores credible:

- **Pre-register the rubrics.** The scoring rubrics and pass thresholds are
  written down and fixed *before* any outputs are scored, and are never
  re-tuned to fit the results.
- **Automate scoring.** Use `evaluators/score_architecture.py` and
  `evaluators/score_debugging.py` wherever possible. They are deterministic, so
  two scorers running them on the same file get identical numbers.
- **Blind where feasible.** When a rubric must be applied manually (e.g. the
  Task 2 launch check), score outputs in a random order without revealing which
  condition each output came from, and record the check before unblinding.

## Statistical rigor

- **Paired design.** All four arms run the same tasks with the same seeds, so
  the natural unit of analysis is the per-pair difference. Pairing removes
  between-task variance that would otherwise inflate the error bars.
- **One primary endpoint.** A3 vs A1 on task quality. Four arms generate many
  possible comparisons; declaring the primary in advance and correcting the
  secondary family (Holm) is what stops the design from becoming a fishing
  expedition.
- **Variance is an endpoint, not a nuisance.** Compare SD between arms with
  bootstrap CIs; normal-theory variance ratio tests do not apply to bounded
  rubric scores.
- **Sample size.** Use at least **n = 5** paired runs per task per arm; more
  with budget. Small samples cannot support strong claims.
- **Effect size.** Report Cohen's d on the paired differences
  (`d = mean(diff) / sd(diff)`), which states the size of the effect in
  standard-deviation units, not just whether it is "present".
- **Confidence intervals.** Report the mean and its **95% confidence interval**
  per condition, plus the mean difference and its CI, rather than bare point
  estimates.
- **Outliers.** Inspect for launch failures, OOMs, and invalid outputs. Decide a
  handling rule (exclude with justification, or run an analysis both with and
  without) *before* running the analysis.
- **What "significant" requires.** A paired t-test or its non-parametric
  equivalent, pre-registered significance level, and enough power for the
  expected effect size. Report the test statistic and the p-value, and do not
  over-interpret a single underpowered run.
- **Honest reporting.** Report failures in the same table as successes. A
  condition that produced more invalid outputs is itself a finding.

## Task design pitfalls

- **Tasks too easy** — ceiling effects: the baseline already scores near-max, so
  there is no room to show an improvement. This risk is much higher with a
  search-enabled control, so run a headroom check on arm A1 before committing to
  the full wave.
- **Tasks a search can simply answer** — if a published recipe matches the task,
  retrieval copies it and the design cannot discriminate. Include tasks that
  force *arbitration* between conflicting sources and *extrapolation* beyond any
  single recipe.
- **Skill-rubric circularity** — rubrics written alongside the skills tend to
  encode what the skills happen to do, which manufactures an effect. Justify
  every criterion against external MoE literature, drop any criterion only the
  skills' phrasing satisfies, and have someone who did not author the skills
  review the rubric. This is the most serious threat to this particular study.
- **Tasks too hard** — floor effects: even skilled agents fail, so scores
  cluster near zero and the test cannot discriminate.
- **Ambiguous scoring** — rubrics that depend on subjective judgment make
  scores noisy and unblinding impossible. Prefer deterministic, rule-based
  scorers.
- **LLM benchmark contamination** — if the task prompts or their reference
  solutions appear in the model's training data, the agent may "remember" the
  answer. Use fresh prompts and periodically rotate the task set.

## Reporting template

The report lives in `benchmarks/results/` and follows the template below,
matching the SKILL.md Expected output section.

Results table format:

| Task | Metric | A0 bare | A1 search | A2 skills | A3 skills+search | A3-A1 | Cohen's d |
|------|--------|---------|-----------|-----------|------------------|-------|-----------|
| task1 | correctness (total /25) |  |  |  |  |  |  |
| task2 | successful launch |  |  |  |  |  |  |
| task3 | accuracy (0-1) |  |  |  |  |  |  |
| task4 | throughput (tok/s) |  |  |  |  |  |  |
| task5 | internal consistency |  |  |  |  |  |  |
| task6 | constraint satisfaction |  |  |  |  |  |  |
| all | score SD (H2) |  |  |  |  |  |  |
| all | total tokens (H3) |  |  |  |  |  |  |
| all | numeric-error rate (H5) |  |  |  |  |  |  |

Analysis-plan structure:

1. Restate research question, hypotheses (H0, H1-H5), and pre-registered rubrics.
2. Describe the runs executed (arm, task, seeds, failures), the retrieval
   configuration and query-log completeness, and any excluded runs with reasons.
3. Report the retrieval-only gain (A1 - A0) before reporting the skill effect.
4. Report the primary endpoint (A3 vs A1): means, 95% CIs, delta, Cohen's d.
5. Report secondary comparisons with the multiplicity correction applied, marked
   as exploratory.
6. Report variance and cost endpoints, which carry the full sample.
7. Conclude honestly: accept/reject H0 per hypothesis with the evidence, state
   where search alone was sufficient, and list limitations.
