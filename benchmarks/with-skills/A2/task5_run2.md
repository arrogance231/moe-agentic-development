# MoE Hyperparameter Decision — 8×80GB GPUs, 1B dense-equivalent

**Scope.** Training setup for a ~1B dense-equivalent MoE on 8×80GB GPUs with standard (non-NVLink-full-mesh) interconnect. Architecture assumptions follow the reference config: 64 experts, top-k, 8 GPUs → EP=8 with 8 experts/GPU and one all-to-all per MoE layer. Memory is *not* the binding constraint (a ~7B-param / 1B-dense-equiv MoE fits 80GB with large headroom); communication cost is.

| Parameter | Chosen value |
| --- | --- |
| Top-k | **2** |
| Capacity factor | **1.25** |
| Auxiliary-loss coefficient | **0.01** |

---

## 1. Top-k = 2

**Conflict weighed.**
- *Top-1* (Switch Transformer, and latency-bound inference deployments): one expert per token, halved dispatch/all-to-all, but a token cannot blend expert knowledge and activated expert capacity is halved — measured quality trails top-2 at equal total experts.
- *Top-2* (GShard, Mixtral; the production default for training): two experts per token, ~2× dispatch, large quality gain from knowledge blending.

**Justification.** This is a training run, not a latency-bound inference deployment, so quality is the objective. The ~2× extra activated-param cost of top-2 is trivially affordable on 80GB GPUs at this scale, and the ~2× dispatch is acceptable across 8 ranks with EP=8. Top-1 at the same 64 experts would waste the expert-count investment. Top-2 also uses the standard convention this budget's capacity-factor and aux-loss values assume.

**Revise if:** the objective shifts to inference latency, or measured step time becomes dominated by all-to-all and a quality hit is acceptable. (A move to top-1 should then revisit CF — top-1 typically pairs with a leaner CF near 1.0.)

## 2. Capacity factor = 1.25

**Conflict weighed.**
- *Conservative/high* (GShard = 2.0): near-zero token dropping under imbalance; but up to ~2× padding waste, larger per-expert buffers, and **2× the all-to-all bytes on every MoE layer** — all paid at every step even when load is balanced.
- *Lean/low* (Switch Transformer ≈ 1.0–1.25 with expert dropout and routing jitter; Mixtral routes with effectively unbounded capacity, no dropping): efficient, minimal padding; but under imbalance, tokens are silently dropped, shrinking the effective batch and starving under-loaded experts of gradients.

**Justification.** 1.25 is the smallest headroom that absorbs realistic short-window imbalance without materially wasting compute, and it matches the safe band (1.0–1.25). The 0.01 aux loss keeps load balanced enough that drops at 1.25 are rare. Because this interconnect is the weakest link in the budget, CF scales dispatch volume directly, so GShard's 2.0 would double all-to-all traffic for no quality gain — deliberately rejected. Memory does not force CF up or down here, so the communication-driven lean value stands.

**Revise if:**
- *Lower to 1.0* when the router analyzer reports overflow/drop rate ≈ 0 and effective experts stay ≥ ~0.5×64 over a window (recovers ~3–8% tokens/sec of padding, per the performance skill).
- *Raise to 1.5* if measured token-drop rate exceeds ~1% (OVERFLOW flags) after already strengthening the aux loss — dropped tokens distort training more than padding waste.

## 3. Auxiliary-loss coefficient = 0.01

**Conflict weighed.**
- *Stronger* (GShard, Switch Transformer = 1e-2): reliable load balance, no collapse; but the aux term competes with the token-prediction objective — ST-MoE observed measurable quality loss at large coefficients.
- *Weaker* (ST-MoE ≈ 1e-3, sometimes paired with router z-loss): preserves routing fidelity; but at 64 experts over only 8 GPUs, an under-powered loss invites collapse → few active experts, idle GPU capacity, and padding.

**Justification.** 64 experts is mid-scale, where collapse is a genuine risk without enforcement, and imbalance has a *direct hardware cost* here (straggler GPUs and idle capacity during all-to-all on a standard interconnect). That biases toward the strong end of the validated band, and 0.01 is the proven GShard/Switch default. It sits inside the recommended 0.001–0.01 window; 0.1 is rejected because it is outside that window and demonstrably distorts routing.

**Revise if:**
- *Increase toward 0.1* (or add router jitter) if normalized entropy collapses / effective experts fall below ~50% of 64 or Gini rises — collapse is worse than mild distortion.
- *Decrease to 0.001* if the run is well-balanced but validation loss is worse than the same config at 0.01, indicating the aux term is distorting routing; pair with a router z-loss if needed.

---

## Validation & cross-check (before committing)

- Recompute parameter/FLOP counts with `tools/moe_calculator.py` and per-GPU memory with `tools/memory_estimator.py`; confirm headroom ≥ 20%.
- Verify config consistency per the training skill's failure modes: EP=8 divides 64 experts; DP×TP×PP×EP = 8; CF/aux/top-k match across the architecture doc, training config, and launch script (a CF mismatch silently changes the effective batch).
- Instrument the first run with the debugging analyzers (`router_distribution.py`, `expert_utilization.py`); the revision triggers above are driven by those measured numbers, not vibes.