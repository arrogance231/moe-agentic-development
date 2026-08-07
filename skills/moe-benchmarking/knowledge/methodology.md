# Research Methodology for LLM-as-Engineer Evaluation

This document is the methodological foundation for the `moe-benchmarking`
skill. It explains why the comparisons are designed the way they are and how to
report results honestly.

## Controlled comparison

The only legitimate way to attribute a performance difference to the MoE skills
is to make the baseline and treatment conditions differ in **nothing but the
skills loaded**:

- **Same model** — the exact same LLM checkpoint and version.
- **Same hardware** — the same GPUs, node topology, and software stack.
- **Same tasks** — the same prompts, inputs, and expected outputs.
- **Same seeds** — the same random seeds where the runtime allows them.

The treatment condition adds the MoE skill suite (the skills under `skills/`)
on top of the baseline agent; everything else is held constant. This isolates
the skill effect: any measured difference can then be attributed to the skills
(or to noise, which the statistics below estimate). If any of these dimensions
differs between conditions, the experiment is confounded and its conclusions
are unsupportable.

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

- **Paired design.** Baseline and treatment runs share the same tasks and the
  same seeds, so the natural unit of analysis is the per-pair difference.
  Pairing removes between-task variance that would otherwise inflate the error
  bars.
- **Sample size.** Use at least **n = 5** paired runs per task per condition;
  more with budget. Small samples cannot support strong claims.
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

- **Tasks too easy** — ceiling effects: baseline already scores near-max, so
  there is no room to show an improvement. Verify the baseline does not saturate
  the rubric.
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

| Task | Metric | Baseline mean ± CI | Treatment mean ± CI | Delta | Effect size (Cohen's d) |
|------|--------|--------------------|---------------------|-------|--------------------------|
| task1 | correctness (total /25) |  |  |  |  |
| task2 | successful launch |  |  |  |  |
| task3 | accuracy (0-1) |  |  |  |  |
| task4 | throughput (tok/s) |  |  |  |  |

Analysis-plan structure:

1. Restate research question, hypotheses (H0/H1), and pre-registered rubrics.
2. Describe the runs executed (condition, task, seeds, failures) and any
   excluded runs with reasons.
3. Report the paired results: per-task means, mean ± 95% CI, delta, and
   Cohen's d.
4. Apply the pre-registered significance test and report the statistic.
5. Conclude honestly: accept/reject H0 with the evidence, and list limitations.
