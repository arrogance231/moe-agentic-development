I'll ground the literature review with a couple of targeted searches before writing the decision document.
# MoE Routing Hyperparameter Decision
**Budget:** 8 × 80GB GPUs (640GB HBM), standard (non-NVLink-full-mesh) interconnect
**Target:** 1B dense-equivalent params

## Decision summary

| Parameter | Chosen value | Literature anchors |
|---|---|---|
| Top-k routing | **1** | Switch (top-1, CF 1.0–1.25) vs. GShard/GLaM/Mixtral (top-2, CF ~1.25–2.0) |
| Capacity factor | **1.25** | GShard CF≈2.0 (top-2) vs. Switch CF 1.0–1.25 (top-1); GLaM CF 1.25 |
| Aux-loss coefficient | **1e-2** | Switch/GShard default ~1e-2; loss-free-balancing ablations span 1e-2–0 |

These values are a coherent package: `top-k = 1` is what makes `CF = 1.25` and `aux = 1e-2` safe; a top-2 router would force a different convention entirely (CF ≈ 2.0 to avoid dropping ~half of tokens), which is exactly the mismatch the prompt flags.

## Budget reasoning (why these interact)

The 1B dense-equivalent MoE (≈4B total params with 8 experts) fits in **~50GB** with optimizer state — memory is not the constraint. The constraints are (1) **all-to-all bandwidth** over standard interconnect and (2) **compute-per-token is small**, which raises the ratio of communication cost to compute cost. Both argue for minimizing dispatch volume and padding. Memory headroom does *not* justify a large capacity factor — padding costs bandwidth, not just memory.

## 1. Top-k routing = 1

**Conflicting positions weighed**
- **Top-2 (GShard, GLaM, Mixtral, DeepSeek)**: the dominant production convention. GShard's ablations show top-2 outperforms top-1 at matched compute; DeepSeek-V3 runs top-8 over 256 fine-grained experts. HF's practical guidance also recommends "top-2 with 1.25 capacity factor" as a starting point.
- **Top-1 (Switch Transformer)**: Switch explicitly found top-1 *at least matches* top-2 quality at fixed compute while halving the dispatch, and is simpler to balance.
- **Expert Choice (EC)**: relevant as evidence that *both* token-choice schemes (top-1 and top-2) need 2–8× overprovisioning to avoid dropping under load imbalance.

**Justification for 1**
- On a standard interconnect, top-1 halves off-GPU all-to-all volume vs. top-2 (~0.88·T·h vs. ~1.5·T·h with E=8, k=1 vs. k=2), which is the dominant cost for a small, comm-bound model. Switch's result (top-1 ≥ top-2 at matched compute) means we give up little quality for that bandwidth.
- With only 8 experts the marginal specialization benefit of a second expert is smaller than at 64–256-expert scale where GShard/DeepSeek's top-2/k advantages were measured.
- Critically, top-1 lets capacity sit at CF≈1.25; top-2 would *force* the CF≈2.0 convention (expected load per expert is k·N/E), doubling padding/comm on the slow interconnect.

## 2. Capacity factor = 1.25

**Conflicting positions weighed**
- **CF = 1.0 (Switch base)**: maximum efficiency; Switch reports good results here but it drops tokens under any routing imbalance, which is common early in training.
- **CF = 2.0 (GShard, EC-CF2)**: GShard's standard for top-2; also the value EC uses to match top-2 compute. Heavy padding cost.
- **CF = 1.25 (GLaM; HF guidance)**: the middle path; GLaM used 1.25 for top-2. Guidance explicitly notes a *smaller* CF is better when all-to-all is slow.
- Token-choice routings generally need 2–8× overprovisioning (EC paper); that argues for CF>1 regardless.

**Justification for 1.25**
- With top-1 the expected load is T/E; CF=1.25 gives 25% headroom to absorb routing imbalance (and the aux loss keeps imbalance bounded), so token dropping stays negligible.
- 1.25 is low enough to keep padding/communication near the efficient floor (Switch showed CF 1.0–1.25 is the sweet spot for top-1), which matters on this interconnect — this is the "use smaller CF when all-to-all is slow" principle.
- It is the point in the 1.0–2.0 range that best balances the two failure modes: dropping (CF too low) and wasted bandwidth on padding (CF too high).

## 3. Aux-loss coefficient = 1e-2

**Conflicting positions weighed**
- **1e-2 (Switch/GShard default)**: the canonical load-balancing weight.
- **1e-1 (upper range)**: used when balance is critical and dropping is catastrophic; risks over-regularizing the router and degrading perplexity (well-documented gradient-interference failure).
- **1e-3 / 1e-4 / 0 (loss-free balancing ablations)**: DeepSeek's aux-loss-free bias scheme uses γ=0.001 with residual α=1e-4; the 1B-scale ablation shows small α → routing collapse, large α → performance loss — the exact dilemma.
- Router z-loss (ST-MoE, 1e-3) is orthogonal stability regularization, not load balancing.

**Justification for 1e-2**
- At 1B scale with E=8 and CF=1.25, the aux loss is the only mechanism keeping expert load inside the capacity slack. Too low (≤1e-3) → routing collapse and sustained token dropping through the 1.25 cap (the failure mode the 1B ablation documents); too high (1e-1) → the router stops specializing and validation loss rises. 1e-2 is the middle of the two-orders-of-magnitude spread that prior work converged on as the balanced operating point, and it is strong enough to hold loads within CF=1.25.

## When I would revise

- **Top-k = 2, CF = 2.0** if profiling shows all-to-all is *not* the bottleneck (8-way scale may be smaller than feared) or interconnect is upgraded to full NVLink mesh — then adopt the GShard convention for the quality headroom.
- **Raise CF → 1.5/2.0** if sustained token-drop rate exceeds ~1% despite aux=1e-2 (early-training imbalance, or batch sizes shrink).
- **Lower CF → 1.0** if padding overhead dominates MoE step time (MFU drop) with near-zero drops observed.
- **Lower aux → 1e-3 (with CF raised to compensate), or adopt aux-loss-free bias balancing (γ≈1e-3, residual α≈1e-4)** if validation loss trails the dense baseline and routing entropy has collapsed (over-regularization). The bias-based scheme is the modern fix at this model scale.
- **Re-tune all three** if the architecture changes materially: more experts (fine-grained, ≥32) makes the top-1 assumption and 1e-2 weight stale, and very large per-GPU batches improve the law-of-large-numbers balance, allowing CF → 1.0.