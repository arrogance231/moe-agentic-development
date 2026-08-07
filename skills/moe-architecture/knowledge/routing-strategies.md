# Routing strategies

How tokens are dispatched to experts, and the levers that control that
dispatch: top-k, capacity factor, and the auxiliary load-balancing loss.

## Token routing overview

In each MoE layer, every token produces a routing logit per expert via a small
router network (a linear projection of the token representation). The router
softmaxes those logits into a distribution over experts, and the top-k experts
are selected. The token is then dispatched to those experts, processed by each
expert's FFN, and the expert outputs are combined back into a single
representation:

```text
token -> router logits (d_model -> num_experts)
      -> softmax -> top-k selection
      -> dispatch (all-to-all) -> expert FFN(s) -> combine -> next layer
```

Two costs are introduced by routing and are central to design decisions:

- **Dispatch / all-to-all cost**: every token crosses to the GPUs hosting its
  chosen experts and back. Grows with `num_experts` and `top_k`.
- **Load imbalance cost**: if experts receive unequal token counts, some GPUs
  idle while others queue, and the capacity factor (below) must absorb the
  imbalance.

## Top-1 routing

**Mechanics**: each token is routed to exactly one expert — the one with the
highest router probability.

**Compute profile**: the cheapest option. One expert FFN per token
(`top_k = 1`), which halves activated expert parameters versus Top-2 and
minimizes all-to-all traffic: one dispatch and one gather per token.

**Quality trade-offs**: each token sees only one expert's FFN, so it can only
draw on that expert's specialization. Knowledge that lives in a second-best
expert is discarded. Empirically, Top-1 lags Top-2 at the same total expert
count and capacity — the second expert adds meaningful capacity and acts as a
soft blend between specializations.

**When to use**: latency- and throughput-bound inference, where halving
activated expert compute and communication is worth the quality loss. Also used
when the router is otherwise high-quality and experts are strongly
specialized. Not the default for training-quality work.

## Top-2 routing

**Mechanics**: each token is dispatched to the two highest-probability experts,
and the two expert outputs are combined by the normalized router weights.

**Why 2 is a common sweet spot**: Top-2 roughly doubles expert capacity per
token (activated params) over Top-1 while keeping the dispatch graph small.
The second expert provides a second opinion per token — a reliable, cheap
quality gain — without the load-balancing difficulty and communication growth
of top-k with k > 2. Most production MoE models (e.g. Mixtral 8x7B) use Top-2.

**Quality vs latency**: quality is close to higher-k variants at a fraction of
the latency. Each increment of k adds dispatch cost and puts more load on the
busiest experts, so k = 2 captures most of the "blending" benefit at the cost
of one extra expert per token. Going beyond k = 2 yields small quality gains
for a linearly growing communication cost.

**When to use**: default for training; default when inference quality matters
and latency budget allows. Use `top_k = 2` unless a profiling run shows the
extra dispatch is the bottleneck.

## Learned and soft routing

**Learned routing**: the router is a trainable linear map on token
representations, learned end-to-end with the model. This is what "routing" now
almost always means — Top-1/Top-2 with a learned router. Heuristic routing
(rules, hashing, domain labels) avoids the router parameters but cannot adapt
to the data and is only used in specialized settings.

**Soft (weighted) routing**: instead of hard top-k selection, every expert
receives a weighted share of every token, or a hard top-k is blended with
soft weights. Variants include:

- **Soft mixing**: all experts process the token, weighted by router
  probabilities — highest quality ceiling, but `num_experts` × compute, so
  only viable with few experts.
- **Weighted top-k**: hard-select top-k, then combine with softmax weights
  (standard practice; the soft weights are what make top-k behave smoothly).
- **Contrastive/auxiliary-regularized routing**: add auxiliary objectives that
  push the router toward specialization (load balancing, routing entropy).

**When to use**: learned routing everywhere by default. Soft weighted combining
inside top-k is standard. Full soft mixing is only reasonable for very small
expert counts where the compute multiplier is affordable. Hard-coded heuristic
routing is a fallback when a learned router cannot be trained (e.g. no
gradients to the router, or a frozen pretrained setup).

## Capacity factor

**Definition**: the maximum number of tokens an expert can process per batch,
as a multiple of the perfectly balanced load:

```text
capacity = ceil(capacity_factor * tokens_per_batch / num_experts)
```

**What 1.0 vs 1.25 means**: at 1.0, each expert can handle exactly its
balanced share of tokens. Routing is never perfectly balanced, so at 1.0 some
expert is always over its share and **drops tokens** (those tokens skip the
expert FFN and pass through with the identity/residual). At 1.25, each expert
can take 25% more than its balanced share, so most imbalance is absorbed and
fewer tokens are dropped, at the cost of up to 25% wasted compute on idle
expert capacity.

**Impact on quality**: dropped tokens lose the expert's contribution to that
layer, which quietly degrades quality without an obvious signal. Higher
capacity factor means fewer drops and better quality but more wasted
compute. This is the direct quality-vs-efficiency dial after top-k.

**Practical values**: 1.0–1.25 is the operating range. Start at 1.25 for
training; tighten toward 1.0 once the auxiliary loss keeps load balanced.
Efficient inference often runs near 1.0 with a strong aux loss.

## Auxiliary load-balancing loss

**Standard form**: the classic load-balancing auxiliary loss is the scaled sum
of per-expert products of the mean router probability and the fraction of
tokens routed to that expert:

```text
aux_loss = aux_scale * num_experts * sum_e (f_e * P_e)

f_e = (tokens routed to expert e) / (total tokens)   # load fraction
P_e = (sum of router probs over e) / (total tokens)  # average gate prob
```

Minimizing the product of `f_e` and `P_e` pushes the router toward uniform
load: any expert either receiving few tokens or low probability is pushed
toward balance. It is differentiable in the router probabilities.

**Typical values**: `aux_scale` between 0.001 and 0.01. Too low and experts
drift into imbalance (the loss is a rounding error in the total loss); too high
and it distorts the routing objective — tokens get routed to balance the
experts rather than to the experts that serve them best, hurting quality.

**Effect on utilization and quality**: a well-tuned aux loss keeps expert
utilization high, which lets the capacity factor stay near 1.0 (fewer drops)
and keeps all-to-all load even across expert-parallel devices. It is the
mechanism that makes large expert counts viable.

## Measuring load balance

Quantify routing balance with these tools:

- **Expert token-count histogram**: per expert, how many tokens were routed in
  a step. A flat histogram is ideal; a long tail is the sign of collapse.
- **Effective number of experts**: `eff_e = (sum f_e)^2 / sum (f_e^2)`, the
  inverse of the token-count Gini-concentration. Ranges from 1 (all tokens to
  one expert) to `num_experts` (perfectly balanced). A healthy router sits
  well above half of `num_experts`.
- **Gini coefficient / entropy of the token-count distribution**: both
  compress the histogram into a single balance score; entropy near
  `log(num_experts)` means balanced, near 0 means collapse.

The moe-debugging analyzers (`skills/moe-debugging/`) compute these from
training logs and flag expert collapse, so pair this skill's design-time
choices with those runtime checks.

## Trade-offs table

| Strategy | Latency | Quality | Communication | Implementation complexity |
| --- | --- | --- | --- | --- |
| Top-1 (learned router) | Lowest | Lower than Top-2 | Minimal (1 dispatch/token) | Low — standard |
| Top-2 (learned router) | Low–medium | High (sweet spot) | 2 dispatches/token | Low — default |
| Higher top-k (3+) | Medium–high | Marginal gain over Top-2 | Grows with k | Low |
| Soft mixing (all experts) | High | Highest ceiling | `num_experts` × token traffic | Medium — heavy compute |
| Heuristic routing | Lowest | Data-dependent | Minimal | Low — no router to train, but no adaptivity |

The dominant decision is usually Top-1 vs Top-2 (latency vs quality), with the
capacity factor and aux loss tuned on top of whichever is chosen.
