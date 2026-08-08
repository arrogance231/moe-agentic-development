Now let me check the workspace for any referenced tools or context files.
# MoE Routing Decision Document

**Budget:** 8 × 80 GB GPUs (non-NVLink-full-mesh interconnect) | **Target:** 1 B dense-equivalent MoE

## 1. Decision summary

| Parameter | Chosen value | Literature range weighed |
| --- | --- | --- |
| Top-k routing | **2** | 1 (Switch, ST-MoE, DeepSpeed-MoE) vs 2 (GShard, GLaM, Mixtral, JetMoE) |
| Capacity factor | **1.25** | 1.0–2.0 (Switch ablation), 1.25 (GLaM), 1.5 (recent fine-grained MoE scaling), ∞/dropless (Mixtral, JetMoE) |
| Auxiliary-loss coefficient | **0.01** | ~0.001 (HF Mixtral default, bias-based balancing regimes) to ~0.1 (strong over-regularization) |

The chosen tuple is the classic **GShard/GLaM-style "balanced top-2" recipe**, biased toward the low end of the capacity-factor range specifically because the interconnect is not a full NVLink mesh: every increment of capacity factor inflates the per-MoE-layer all-to-all dispatch volume, which is the dominant communication cost on this topology.

## 2. Top-k = 2

### Conflicting positions weighed
- **Top-1 (Switch Transformer, ST-MoE, DeepSpeed-MoE).** Halves dispatch/all-to-all volume per MoE layer vs top-2 — a real win on a bandwidth-constrained interconnect. Single-expert routing is simpler and empirically scales to trillion-parameter models. However, Mixtral's matched-compute comparison found top-2 **materially outperforms top-1**, and top-1 concentrates all routing variance on one expert, raising overflow risk (which in turn demands a higher capacity factor).
- **Top-2 (GShard, GLaM, Mixtral 8x7B, JetMoE).** The training-quality default. Gradients flow through two experts per layer, which empirically suppresses router collapse; adds knowledge blending. Cost is ~2× dispatch volume vs top-1.

### Justification
At 1 B dense-equivalent, per-token expert compute is already small, so the absolute all-to-all cost of the second expert is modest — a 1 B-scale expert FFN is tiny next to a 7 B-scale one (Mixtral 8x7B). The documented quality gain (and collapse-resistance) of top-2 dominates the added dispatch traffic at this scale. On 8 ranks, the top-2 all-to-all is a bounded, 8-way exchange; it is a solvable engineering cost, not a wall. Top-1 would be the choice only if the run were demonstrably communication-bound, which the interconnect makes plausible but not given at this scale.

### Revision conditions
Switch to top-1 **if** a profiler shows the all-to-all share of step time consistently exceeds ~25–30% (measured with `throughput_profiler.py` / step-time breakdown), **and** a head-to-head top-1-vs-top-2 run at fixed seed and equal compute shows the loss gap is negligible. Top-1 is also the default if the goal pivots to latency-bound inference.

## 3. Capacity factor = 1.25

### Conflicting positions weighed
- **1.0.** Zero padding waste at balanced load, but the Switch ablation measures **10–20% dropped tokens** under realistic imbalance — the largest drop rate of any candidate.
- **1.25.** Switch ablation: **<5% drops**, ~20% worst-case padding waste; the GLaM default; the value the Switch authors recommend.
- **1.5.** Used by recent fine-grained MoE scaling work (CF 1.5, aux 1e-2, z-loss 1e-3); buys margin at ~33% padding waste.
- **2.0.** Top of the range: minimal drops but ~50% padding waste at balanced load, and it doubles dispatch volume per MoE layer on an interconnect that is already a constrained resource.
- **∞ / dropless (Mixtral, JetMoE).** No capacity constraint at all; only viable with specialized block-sparse kernels and no fixed-buffer dispatch, which most frameworks do not expose cleanly.

### Justification
Padding waste at balanced load is ≈ `1 − 1/CF` (1.25 → ~20%, 2.0 → ~50%), and on non-full-mesh interconnect every padded slot still crosses the network. Since aux loss 0.01 (below) keeps the router near-balanced, 1.25 buys the documented <5% drop rate with the lowest padding/dispatch overhead among the safe choices. 1.0 is rejected because the router is never perfectly balanced and 10–20% drops are an unacceptable quality hit; 1.5–2.0 are rejected because they waste compute and, more importantly for this budget, inflate all-to-all volume on a bandwidth-constrained topology for margin the aux loss already provides. 80 GB × 8 = 640 GB gives no memory pressure, so nothing pushes the factor upward for memory reasons.

### Revision conditions
- **Raise to 1.5** if the measured per-expert overflow fraction exceeds ~5% despite the aux loss (imbalance persisting), e.g. measured with `router_distribution.py`'s `OVERFLOW` flag.
- **Lower toward 1.0** once measured drops are <1% **and** the run is compute- or communication-bound; on this interconnect, dropping the factor also cuts dispatch volume directly.

## 4. Auxiliary-loss coefficient = 0.01

### Conflicting positions weighed
- **~0.001.** The HF Transformers `router_aux_loss_coef` default for Mixtral, and the weight used when a *separate* balancing mechanism (DeepSeek-V3's bias-based EMA) does the heavy lifting. At this scale, reviews document 0.001 as under-regularizing — collapse risk on long runs.
- **0.01.** The consensus across GShard, Switch, ST-MoE, and the 2025 fine-grained MoE scaling recipe.
- **~0.1.** The top of the two-order-of-magnitude range. Reviews uniformly flag it as over-regularizing: the router is forced toward uniform use and loses the ability to specialize (the "fake distribution" pitfall), and the aux gradient distorts the language-modeling objective.

### Justification
0.01 sits in the middle of the 0.001–0.1 range and is the value used by the papers that established each of the other two parameters (GShard/Switch/ST-MoE). At 1 B scale the router is small and its gradients are noisy early in training, so collapse is a live risk; 0.01 provides enough balancing pressure to keep the effective number of experts high without the specialization-killing uniformity that 0.1 produces. It is also correctly *paired* with the other two choices: it justifies the low 1.25 capacity factor (a balanced router does not need 1.5–2.0 slack) and it is the standard companion to top-2.

### Revision conditions
- **Raise toward 0.02–0.05** if effective experts fall below ~0.5 × `num_experts` (collapse signal from `router_distribution.py` entropy/Gini/effective-expert metrics).
- **Lower toward 0.001–0.005** if the aux loss visibly distorts the routing objective — e.g. the router is uniformly balanced but loss/perplexity regresses, or per-expert entropy is at its ceiling while quality degrades. Also the natural value to use if the team later adopts bias-based (aux-loss-free) balancing.
- Add a router **z-loss at 1e-3** as a stability complement; it is cheap and documented (ST-MoE) to remove logit-growth instability at no quality cost.

## 5. Coupling note

These three values are not independent. The tuple (top-2, 1.25, 0.01) is internally consistent: top-2 supplies the quality and collapse-resistance, 0.01 keeps the router balanced enough that 1.25's slack is sufficient (so 1.5–2.0 is unnecessary), and 1.25 keeps the all-to-all volume that top-2 introduces as low as is safely possible on a non-full-mesh interconnect. Change one and the others should be revisited: a move to top-1 (to halve dispatch) should first re-test the capacity factor, since top-1's higher per-expert variance raises overflow risk.

## 6. Validation checklist before locking in

1. Profile step time and confirm all-to-all is not the dominant term (≥1–2 MoE layers' worth of dispatch) on the target interconnect.
2. After ~5–10k steps, verify effective experts ≥ 0.5 × `num_experts` and per-expert counts within ~2× of the mean (`router_distribution.py`).
3. Verify per-layer overflow fraction < 5% (capacity factor holds; `OVERFLOW` flag clean).
4. Confirm z-loss keeps summed router logits bounded if added.

All three chosen values are defaults with strong precedent; the decision document's purpose is that they are *specific, justified, and coupled* — and each has an explicit, measurable trigger for revision.