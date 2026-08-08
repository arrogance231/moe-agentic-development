I'll ground the literature claims with a quick search, then produce the decision document.
# MoE Routing & Load-Balancing Decision — 8×80GB GPUs, 1B Dense-Equivalent

## Decision summary

| Parameter | Chosen value |
| --- | --- |
| top-k | **2** |
| capacity factor | **1.0** |
| auxiliary-loss coefficient | **0.01** |

This is the **Mixtral/GShard convention** (top-2 + cf=1.0 + drop-on-overflow), as opposed to the **Switch convention** (top-1 + cf=1.25). Memory is not the binding constraint — 640GB HBM is ample for a 1B dense-equivalent MoE (64 experts, top-2 stays well under budget). The decision is driven by *communication* (standard, non-full-mesh interconnect) and *quality-per-FLOP* at 1B scale.

## 1. top-k = 2

**Positions weighed**
- **top-1** (Switch Transformer, ST-MoE, DeepSpeed-MoE): one expert per token → halves all-to-all dispatch volume and expert compute. But Switch's own analysis notes top-1 can underperform top-2 at matched compute; it becomes robust only when wrapped in a capacity buffer + balancing loss.
- **top-2** (GShard, Mixtral, DeepSeek-MoE, GLaM): blends two expert views, ~2× activated expert capacity per token; the standard for training-quality work.

**Justification.** Mixtral empirically found top-2 materially outperforms top-1 at matched compute budgets, and at 1B scale quality-per-FLOP is the goal. The interconnect concern is bounded here: 8 ranks, 8 experts/rank at EP=8, one all-to-all per MoE layer. top-2 doubles dispatch volume versus top-1, but on an 8-rank ring that is a modest, fixed cost, not a scaling wall. This also matches the canonical 64-expert / top-2 / 8-GPU config for this benchmark family.

## 2. Capacity factor = 1.0

**Positions weighed**
- **1.0** (GShard, Mixtral, DeepSeek-MoE): exact slots per expert; maximum compute/communication efficiency; drops tokens when routing is imbalanced. Presupposes the aux loss keeps load near-uniform.
- **1.25** (Switch default for non-scale runs, GLaM): absorbs imbalance with ~1% wasted compute/padding; Switch data shows <5% drops at 1.25 vs 10–20% at 1.0.
- **2.0** (upper literature bound): minimal drops but wastes ~2× padding and — decisively here — ~2× all-to-all dispatch volume.

**Justification.** Dispatch/receive volume scales linearly with capacity factor, and on a non-full-mesh interconnect all-to-all is the scarce resource. cf=1.0 uses every dispatched byte for real tokens, and the aux loss is tasked with keeping routing balanced enough that overflow stays rare. We buy robustness from the *loss* (which is nearly free on this hardware) rather than from *padding* (which is expensive here). The revision rule below absorbs the early-training drop risk.

## 3. Auxiliary-loss coefficient = 0.01

**Positions weighed**
- **0.001** (low end): minimal distortion of the routing objective, but too weak to hold balance.
- **0.01** (GShard and Switch production default): strong enough to prevent collapse without dominating the loss.
- **0.1** (upper end, GShard's own stated range): maximally strong balancing; but the Expert Choice work documents that large aux weights favor "balanced but less effective routing" — over-uniform experts lose specialization.

**Justification.** Choosing cf=1.0 makes us *structurally dependent* on the aux loss: it is the only mechanism keeping drops and collapse in check at exact capacity. 0.01 is the middle of the two-orders-of-magnitude spread that is known to work in production (GShard/Switch both used it), strong enough to prevent collapse and hold drops low at cf=1.0, yet an order of magnitude below the level shown to distort routing. 0.001 would leave cf=1.0 fragile; 0.1 is unnecessary risk.

## Interaction

top-2 + cf=1.0 + aux=0.01 trades ~2× dispatch volume against per-token expert capacity and quality; top-1 + cf=1.25 trades the reverse. We take the former because 8-rank all-to-all is affordable while 1B-scale quality is not something to give away.

## Revision conditions

- **Raise cf to 1.125–1.25** if measured token-drop/overflow rate exceeds ~2–5% during early training (checked with a per-expert distribution analyzer), or if effective experts fall below 0.5×N. Revert toward 1.0 once the router stabilizes.
- **Raise aux loss toward 0.03–0.1** only if collapse persists after cf=1.25; **lower toward 0.001** (or adopt loss-free bias balancing à la DeepSeek-V3) only if downstream quality regresses while balance is healthy.
- **Drop to top-1** only if all-to-all dominates step time (>~30–40%) and throughput becomes the binding constraint; accept the quality cost and move to the Switch cf=1.25 convention.
- **Hardware changes:** full-mesh NVLink lowers communication cost → top-2/cf=1.0 become more comfortable; scaling GPU count on the same interconnect grows all-to-all and pushes toward top-1 / higher EP.