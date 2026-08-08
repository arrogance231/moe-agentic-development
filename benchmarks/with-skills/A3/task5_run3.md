I'll verify the key literature values I want to weigh before writing the decision document.
# MoE Routing Hyperparameter Decision
**Target:** 1B dense-equivalent MoE · **Hardware:** 8× 80GB GPUs, standard (non-full-mesh) interconnect · **Use case:** training-quality benchmark (not latency-bound inference)

## Assumed architecture context
The skills' reference config for this scale (1B dense-equivalent, 8 GPUs) uses **64 experts** with expert FFNs sized to keep total params ~2–4B. On 80GB/GPU, memory is not binding; the binding constraint is **all-to-all communication over a standard interconnect**. This drives every choice below.

## Decision summary

| Parameter | **Choice** | Literature range | Rationale in one line |
| --- | --- | --- | --- |
| Top-k | **2** | 1 vs 2 | Quality + a second-slot safety valve that makes token drops rare |
| Capacity factor | **1.25** | 1.0–2.0 | Absorbs early-training imbalance at only +25% dispatch volume |
| Aux-loss coefficient | **0.01** | 0.001–0.1 | The one value validated by Switch's own sweep: balances fast without distorting the LM objective |

The three are mutually reinforcing and must not be tuned independently: per-expert buffer is `cf × tokens × top_k / E`, so top-k and cf are coupled across conventions.

---

## 1. Top-k = **2**

**Positions weighed**
- **Top-1** — Switch Transformer (Fedus et al., 2021) adopted it explicitly for simplification: half the dispatch/all-to-all volume and simpler gradients, with quality reported as at least matching Top-2 at equal FLOPs. DeepSpeed-MoE and DeepSeek (fine-grained + shared experts, small effective k) also route Top-1/≈Top-1.
- **Top-2** — GShard (Lepikhin et al., 2020), GLaM, and Mixtral (Jiang et al., 2024) use it as the training default for knowledge blending and robustness. Under capacity pressure the second slot is a *safety valve*: a token overflowed on its first expert can still be served by its second, keeping the "fully dropped" (identity pass-through) rate near zero — something Top-1 cannot do at any capacity factor.

**Justification (not just a range):** This is a training-quality run, so the 2× dispatch cost of Top-2 is worth paying; nothing in the brief is latency-constrained. At 1B dense-equivalent with 64 experts, per-expert activated capacity per token is small, and Top-1 would halve it — the worst case for a 64-expert model. The communication penalty is contained because I pair Top-2 with cf = 1.25 (not 2.0) and because the second slot reduces the drop rate that would otherwise force a higher capacity factor. Note the convention trap in the literature: a Top-2 system at cf = 1.25 gives each expert ~2.5× the slots of a Top-1 system at cf = 1.25, so raw "cf" values are only comparable *within* a top-k convention.

**Revision conditions** — switch to Top-1 if (a) the profiler shows all-to-all/dispatch is >~30% of step time and the quality target is already met; (b) the deployment becomes latency-constrained; (c) router analysis shows second-expert gate mass is negligible across layers (`gate_2 ≪ gate_1`) — the router is effectively Top-1 already, so make it explicit and halve communication.

---

## 2. Capacity factor = **1.25**

**Positions weighed**
- **1.0** — Switch authors report better speed-quality at "lower capacity factors (1.0, 1.25)" and recommend shrinking cf in memory-scarce regimes; the NeMo Mixtral reference config uses 1.0. No padding waste, minimal dispatch buffer.
- **1.25** — GShard and GLaM defaults; the HuggingFace MoE guidance calls "top-2 routing with 1.25 capacity factor" a good starting point. 25% slack absorbs routing variance without much wasted compute.
- **1.5–2.0** — Community/Switch follow-up guidance: raise when drop rates run high; cf = 2.0 "essentially eliminates dropping at the cost of twice the memory" and is common in eval-time configs.

**Justification:** cf = 1.0 gives *zero* slack, and routing is inevitably imbalanced in early training before the aux loss converges — so drops occur, starve the overflowed experts' gradients, and self-reinforce the imbalance. cf = 2.0 carries a 60% larger dispatch buffer than 1.25, and on a standard interconnect that volume directly buys wall-clock time for zero quality benefit once the aux loss holds drops <1%. cf = 1.25 sits at the point where the *fully-dropped-token* rate stays ≈0 (GShard/GLaM production experience) for only +25% dispatch volume. Because one cf must be fixed for the whole run, 1.25 hedges the early-training window.

**Revision conditions** — raise to 1.5–2.0 if, after warmup, assign-drop / fully-dropped rate stays above 1–2% (`OVERFLOW` flag from `router_distribution.py`) — either the aux loss is under-controlling imbalance or batches are too small for the routing variance. Lower to 1.0 once the load is measured balanced (effective experts ≥ 0.8×n, drop rate ≈ 0) *and* the profiler shows padding/communication dominating step time.

---

## 3. Aux-loss coefficient = **0.01**

**Positions weighed**
- **0.001** — DeepSeek-MoE/DeepSeek-V2 use 1e-3, but only because they add device-level and communication balance losses plus per-sequence aux (and later move to aux-loss-free bias balancing) — not comparable to a single Switch-style term. HuggingFace's `router_aux_loss_coef` defaults (Switch, Mixtral) are 0.001.
- **0.01** — Switch Transformer's own sweep (α from 1e-1 to 1e-5 in powers of 10) selected 1e-2: "sufficiently large to ensure load balancing while small enough to not overwhelm the primary cross-entropy objective." GShard, GLaM, and NeMo's guidance (1e-2 "a good start") converge here.
- **0.1** — upper end of the reported range; forces near-uniform routing and suppresses the expert specialization that 64 experts exist to provide.

**Justification:** At this scale each expert sees far fewer tokens per step than at production scale, so the router is *more* prone to imbalance and collapse; 1e-3 risks under-balancing and would inflate drops under the cf = 1.25 policy. 0.1 distorts the routing objective and caps specialization — the opposite of what a 64-expert model buys. 1e-2 is the single value directly validated by Switch's sweep for exactly the goal here (balance quickly, don't interfere with the LM loss), and it is the production-verified value for the Top-2/cf-1.25 combination. With Top-2 + cf 1.25, α = 1e-2 keeps effective experts high and drops ≈ 0.

**Revision conditions** — raise toward 0.02–0.1 if, after warmup, effective experts < 0.5×n or Gini stays high (`router_distribution.py`), or the drop rate stays elevated. Lower to 0.001–0.003 if the aux term becomes a large share of total loss or the routing distribution goes near-uniform while validation loss stalls (over-regularization suppressing specialization).

---

## Coherence check
Top-2 makes drops rare (second-slot valve); cf = 1.25 keeps dispatch volume within +25% of the theoretical minimum; α = 0.01 keeps the load balanced so the 1.25 slack is rarely consumed. The rejected alternatives interact badly on this hardware: α = 0.001 would force cf ≈ 1.5–2.0 to stop drops (≈2× the all-to-all volume on a standard interconnect), and Top-1 would need cf near 2.0 to hold drops low — both strictly worse for this budget than the chosen combination.

## Monitoring plan
Track per the `moe-debugging` analyzers each run: per-expert counts, effective experts, Gini, assign-drop/fully-dropped rate, and the all-to-all share of step time (`throughput_profiler.py`). Revisit any choice only when a metric crosses the revision thresholds above — not preemptively.