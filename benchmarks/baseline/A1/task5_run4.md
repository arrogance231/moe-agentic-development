I'll research the current literature to ground this decision, then produce the document.
Let me verify a couple of specifics (GShard/ST-MoE conventions) before writing the document.
# MoE Hyperparameter Decision — 8×80GB GPUs, 1B Dense-Equivalent Target

**Scope:** Single architectural decision for a ~1B dense-equivalent sparse MoE (e.g., 8 experts/layer, top-k routing, feed-forward-only MoE layers) trained on 8×80GB GPUs with standard (non-NVLink-full-mesh) interconnect.

---

## 1. Decisions (summary)

| Parameter | **Chosen value** | Range published in the literature |
|---|---|---|
| Routing granularity (top-k) | **top-1** | top-1 (Switch, ST-MoE) vs. top-2 (GShard, GLaM, Mixtral) |
| Capacity factor (CF) | **1.25** (train), 2.0 (eval, optional) | 1.0–2.0; GShard 2.0, Switch 1.0–1.25, ST-MoE 1.25 |
| Auxiliary load-balancing loss coefficient (α) | **0.01** | 0.001–0.1 across works |

**Context that drives everything else:** a 1B dense-equivalent MoE (total params ≈ 2–8B) fits entirely within the 80GB of a *single* GPU. This is exactly the ST-MoE-L regime, which "fits within accelerator memory" and therefore "does not require model parallelism." On this budget we run **pure 8-way data parallelism (no expert sharding)**, which removes cross-GPU all-to-all from the MoE critical path. The standard interconnect therefore does *not* constrain capacity factor or top-k — those should be chosen for compute/quality Pareto efficiency, with communication as a secondary (robustness) concern. Had we instead been forced to shard experts (i.e., a much larger model), the same choices remain the conservative ones: top-1 and CF 1.25 are exactly the settings that minimize all-to-all volume on slow interconnect.

## 2. Conflicting positions weighed

**Top-k.**
- **Top-1 (Fedus et al., 2021 — Switch):** single-expert routing plus a capacity buffer, the Switch load-balancing loss, and an fp32 router matches top-2 quality at 4–7× speedup, and lets you double the number of experts for the same per-token compute. Used by DeepSpeed-MoE and ST-MoE (with z-loss).
- **Top-2 (Lepikhin et al., 2020 — GShard; Du et al., 2021 — GLaM; Jiang et al., 2024 — Mixtral):** two experts per token give smoother, more robust routing and better per-token quality; top-2 is the production standard (Mixtral 8×7B), with the GShard line pairing top-2 with a **larger capacity convention (CF 2.0)** to absorb two assignments per token.
- ST-MoE's head-to-head at this exact scale (Table 8/18) shows the top-1 vs. top-2 gap is tiny (≈0.003 val-loss) — top-1 slightly better at CF 1.0, top-2 slightly better at CF 1.25 — and that the difference is within the hardware-dependent Pareto noise.

**Capacity factor.**
- **Low (1.0–1.25):** Switch reports good performance at low CF and recommends 1.25 as Pareto-efficient; ST-MoE measures +7% step time at 1B going 1.25→2.0 for negligible quality gain; ST-MoE fine-tuning quality survives 10–15% token drop.
- **High (2.0, GShard/GLaM):** fewer dropped tokens and slightly better quality (ST-MoE Table 8: CF 2.0 → −1.360 vs. −1.375 at 1.25), but strictly more memory/padding and more all-to-all traffic. ST-MoE explicitly states slow all2all/allreduce favors smaller CF; on fast interconnect, larger CF and larger k become optimal.

**Auxiliary-loss coefficient.**
- **Low (≈0.001):** under-regularizes; routing collapse and hot/idle experts return on long runs.
- **Canonical (0.01):** the replicated default across Switch, GShard, GLaM, and ST-MoE; balances "enough balancing pressure" against "forcing uniform noise."
- **High (≈0.1):** over-regularizes — the router flattens to near-uniform and experts stop specializing, trading balance for task loss.
- **Counter-position:** auxiliary-loss-free load balancing (DeepSeek-V3 / Wang et al., 2024) removes the coefficient entirely via per-expert routing biases. Rejected here: it is a different balancing mechanism, and the brief requires an auxiliary-loss coefficient; α=0.01 keeps the option open to later migrate to loss-free bias updates without changing the rest of the design.

## 3. Justification of the chosen values

**top-1.** The binding resource on this budget is *per-token compute*, not memory (640GB ≫ a 2–8B model). Top-1 halves expert FLOPs per token relative to top-2, so for a fixed 1B dense-equivalent compute envelope we can hold **roughly twice the total expert capacity** — more stored knowledge per FLOP, which is the actual MoE bargain. Memory abundance makes that larger parameter footprint free. Switch demonstrated top-1 matches top-2 once the capacity buffer, the balancing loss, and fp32 routing are in place, and ST-MoE at this exact 1B scale confirms the top-1/top-2 difference is within noise at CF 1.25. Top-1 also halves all-to-all traffic and router traffic if expert parallelism is ever introduced on this interconnect, making it the conservative choice under the hardware given.

**CF = 1.25 (train), 2.0 (eval).** The Pareto-efficient point identified by both Fedus et al. and Zoph et al. at this scale. CF 1.0 cannot absorb early-training imbalance or batch-composition spikes (ST-MoE Table 5: with aux loss, CF 0.75 → 10.6% tokens dropped vs. 0.3% at CF 1.25); CF 2.0 costs +7% step time for a gain inside the noise (Table 9). 1.25 keeps drop rate < ~1% once the aux loss converges, which is the threshold at which dropped tokens become quality-relevant. Because eval batches are short and the model is small, a separate eval CF of 2.0 (ST-MoE's recommendation) buys additional eval quality at negligible cost.

**α = 0.01.** The most replicated coefficient in the literature (Switch, GShard, GLaM, ST-MoE), and the only one with reported failure modes bracketing it on both sides (KempnerForge: 0.001 → collapse risk, 0.1 → over-regularization). At 1B with 8-way data parallelism, the per-GPU token batch feeding the load-balance estimator is small, so the fᵢ·Pᵢ estimate is noisier than at frontier scale; a coefficient at the tested default (rather than at 0.001) provides the balancing margin this noisiness demands without sacrificing specialization. It is complemented by router **z-loss at 1e-3** for bf16 logit stability (ST-MoE), which the α choice assumes but does not replace.

## 4. When I would revise the choice

Instrument drop rate, padding efficiency, aux-loss trend, per-expert token histogram, and router entropy per layer from step 1.

| Trigger | Revision |
|---|---|
| Drop rate > 5% after warmup, or val-loss regresses vs. a 1B dense baseline | Raise CF → 1.5; if drops persist, fall back to **top-2 with CF 2.0** (GShard convention) — the +7–14% step time is affordable at this scale |
| Drop rate stays < 1% and measured padding waste > ~25% of expert FLOPs | Lower CF → 1.0 (reclaim compute); consider dropping the aux loss in fine-tuning, where ST-MoE shows up to 10–15% drop is harmless |
| Routing collapse: aux loss rising, dead experts, entropy collapsing | Raise α → 0.05–0.1 and/or raise z-loss coefficient; verify α isn't dwarfing task loss |
| Aux loss dominates task loss or eval degrades from forced uniformity | Lower α → 0.001; as a structural alternative, migrate to auxiliary-loss-free bias balancing |
| Model target grows beyond the single-GPU-memory regime (≈30B+), forcing expert sharding | Re-derive: sharding makes all-to-all volume the first-class cost, favoring **top-1 + lower CF**; if interconnect is upgraded to full NVLink mesh, re-opt toward **top-2 + CF 2.0** (fast all2all shifts the Pareto frontier — ST-MoE, §6.1) |
| Production traffic is more homogeneous than the training mix | Raise serving-time CF (or use a dropless block-sparse kernel, e.g. MegaBlocks) — homogeneous batches overload the same experts and drop more |

---

**Bottom line:** top-1 routing, capacity factor 1.25 (train) / 2.0 (eval), auxiliary-loss coefficient 0.01, with router z-loss 1e-3. This is the setting that is Pareto-efficient at the 1B scale specifically, fits the memory-rich but compute- and interconnect-constrained budget, and degrades gracefully in either direction if measured drop rates or loss trends deviate from the ranges above.