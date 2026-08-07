# Router collapse and expert imbalance

Why MoE routers collapse onto a few experts, how that shows up in the
analyzers, and the load-balancing fixes that restore balance.

## Router collapse

**Mechanism.** The router is a learned linear map that softmaxes token
representations into a distribution over experts. Collapse is a
self-reinforcing feedback loop: popular experts receive more tokens, so they
receive more gradient, so they get better at predicting tokens, so they get
even more tokens. Without a countervailing force, the routing entropy shrinks
until almost every token goes to a tiny subset of experts. In the entropy
language used by `analyzers/router_distribution.py`, normalized entropy
`H = -sum(p_i ln p_i) / ln(n)` slides from ~1.0 (uniform) toward 0.0, the
effective number of experts `exp(H_raw)` falls toward 1, and the top-expert
share climbs past 0.5.

**Why it hurts.** A collapsed router nullifies the expert-count investment:
most expert parameters are dead weight, the few active experts overfit
(repeatedly see the same tokens and dominate the gradient), and the capacity
factor cannot absorb the resulting concentration — tokens get dropped from
over-capacity experts. The symptom on the loss curve is a plateau: the model
keeps training but stops improving because capacity is wasted.

**How to measure.** Run `analyzers/router_distribution.py` on per-expert
token counts. The `COLLAPSED` flag fires when the top-expert share exceeds
0.5. Watch the normalized entropy (should stay comfortably above ~0.6), the
effective number of experts (should stay above `0.5 * n`), and the Gini
coefficient (should stay below ~0.3).

## Expert imbalance

**Causes.** Imbalance is the milder sibling of collapse and usually precedes
it: the aux loss is too weak for the data bias, the LR is high enough to let
the router drift, gradient noise under-trains unpopular experts, or the data
itself is skewed so some experts are genuinely more relevant. A high capacity
factor can mask imbalance by absorbing the skew without visible token drops,
which is why a healthy-looking step time and loss can hide a badly skewed
router.

**Measurement.** The same `router_distribution.py` metrics detect imbalance
before it becomes collapse: the `IMBALANCED` flag fires when Gini exceeds 0.3
or the effective number of experts falls below `0.5 * n`. `expert_utilization.py`
adds the operational view — utilization %, the max/min skew, and which
experts sit at or over their balanced capacity. The distinction matters for
the fix: imbalance responds to load-balancing losses and jitter, whereas full
collapse may additionally need a capacity-factor or LR correction.

**Relationship to collapse.** Imbalance is a gradient on the same axis: the
same feedback loop operating at a smaller amplitude. Imbalance is recoverable
with jitter and a stronger aux loss; collapse is what you get when neither is
present and the loop has run long enough. Treat an `IMBALANCED` flag as a
pre-collapse warning rather than a separate failure.

## Fixes

**Aux load-balancing loss.** The standard form penalizes the scaled product of
load fraction and mean routing probability per expert:

```text
aux_loss = aux_scale * num_experts * sum_e (f_e * P_e)
```

Typical `aux_scale` values are **0.001–0.01**. Too low and the term is a
rounding error against the task loss; too high and it distorts routing toward
balance at the expense of quality. When the router collapses, move to the top
of the range and, if the standard form is too weak, switch to a stronger
variant (e.g. a loss on token counts rather than soft probabilities, or a
differentiability-aware load-balancing form).

**Jitter noise.** Adding noise to router logits at train time forces the
router to explore experts it would otherwise never touch, so unpopular experts
keep receiving gradient. Applied to the logits (not the probabilities) with
the noise annealed or scaled with the LR. A fixed seed makes jitter
reproducible for ablations.

**Expert dropout.** Randomly dropping whole experts from a token's candidate
set during training has a similar effect to jitter: it prevents a small
coterie from monopolizing gradient. Use sparingly, since it also reduces
effective capacity.

**Capacity factor.** Keep it ≥ 1.0 (1.0–1.25) so under-loaded experts keep
receiving tokens and their gradient stays alive. A factor below 1.0 drops
tokens from exactly the experts that are over-loaded, starving the rest and
accelerating collapse.

**LR control.** Collapse is sometimes a symptom of an LR too high for the
router to stay balanced. Lowering the LR or lengthening warmup gives the
router time to differentiate experts without one runaway expert.

**Ablation methodology.** All fixes are testable with short runs:
- Fix the seed and data so the only variable is the fix.
- Compare effective-expert counts, Gini, and loss at the same step count
  (e.g. aux 0.01 vs 0; jitter on vs off).
- Confirm with the analyzers, not the loss alone — a better loss with a
  collapsed router still fails the effective-experts check.

## Measurement tools

- **`analyzers/router_distribution.py`** — the primary collapse/imbalance
  detector. Run on an `expert,count` (or `expert,probability`) CSV. Read the
  normalized entropy, effective experts, Gini, top-expert share, and the
  `COLLAPSED`/`IMBALANCED`/`OVERFLOW` flags. `OVERFLOW` (fraction of experts
  above `total/n × capacity_factor` > 0.1) tells you tokens are being dropped
  at the current capacity factor.
- **`analyzers/loss_analyzer.py`** — confirms the collapse symptom on the loss
  side: `PLATEAU` fires when the tail slope is ~0 and the final loss sits
  well above the curve minimum, which is the classic collapse signature.
- **`analyzers/expert_utilization.py`** — operational view: utilization %,
  max/min skew, the per-expert at/over-capacity table, and `OVERFLOW` when the
  over-capacity experts hold more than 10% of tokens.
- **`tools/diagnosis_report.py`** — composes all three into one report and
  maps the strongest flag to a failure class.

**Interpretation thresholds.** Healthy: entropy > 0.6, effective experts >
`0.5 * n`, Gini < 0.3, utilization ≥ 70%, overflow fraction ≤ 0.1. Collapse:
`COLLAPSED` (top share > 0.5). Imbalance: `IMBALANCED` (Gini > 0.3 or
effective experts < `0.5 * n`) or `UNDERUTILIZED` (utilization < 70%). A
`PLATEAU` flag alongside a collapsed router is the confirmation that the
imbalance is costing you loss.
