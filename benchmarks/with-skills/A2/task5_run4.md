I'll ground the decision in the published literature and the skill formulas. Let me verify the exact literature values I'll cite (Switch, GShard, Mixtral, DeepSeek conventions), since the "conflicting positions" section must be accurate.
Let me verify the GShard and DeepSeek-MoE conventions I plan to cite.
I have verified the literature values (Switch α=0.01 with a 10⁻¹–10⁻⁵ sweep; Mixtral released config 0.02 + z-loss; DeepSeek 0.01 at 2B / 0.001 at 16B; GShard top-2 with capacity ~2N/E; GLaM cf 1.25; Switch capacity ablations at 1.0/1.25/2.0). Here is the decision document.

---

# Decision Document — MoE routing hyperparameters
## Budget: 8× 80GB GPUs, standard (non-full-NVLink-mesh) interconnect · Target: 1B dense-equivalent

### Decisions

| Parameter | **Chosen value** |
| --- | --- |
| Top-k routing | **2** |
| Capacity factor | **1.25** |
| Auxiliary-loss coefficient | **0.01** |

This is a mutually consistent triple — the GShard-style configuration (top-2, ~1.25 capacity, weak-to-moderate aux loss) — and each value is chosen for a specific, quantified reason, not a range.

### Grounding architecture (used for the arithmetic below)

24 layers · d_model 1536 · FFN 6144 · 16 experts · top-2 · vocab 32000 · seq 2048. Figures hand-computed with the `moe-architecture` skill formulas; re-validate with `tools/moe_calculator.py` where available.

| Component | Dense | MoE (16 exp) | Activated (top-2) |
| --- | --- | --- | --- |
| Attention (all layers) | 226.5M | 226.5M | 226.5M |
| Expert FFNs (all layers) | 679.5M | 10,871.6M | 1,359.0M |
| Embedding | 49.2M | 49.2M | 49.2M |
| **Total** | **955.2M** | **11,147.4M** | **1,634.7M** |

Param ratio ≈ 11.7×; activated ≈ 1.71× dense; FLOPs/token ≈ 10.1 GFLOPs (~1.7× dense). Estimated steady-state memory ≈ 25–30 GB/GPU incl. activations → **memory is not the binding constraint** (this fact drives the capacity-factor choice below).

---

### 1. Top-k = 2

**Conflicting positions weighed.**
- **Top-1** (Switch Transformer; ST-MoE; DeepSpeed-MoE): halves dispatch/all-to-all volume vs top-2, simplifies routing and gradients. Switch frames this as its headline simplification and claims a speed–quality win at matched compute.
- **Top-2** (GShard; GLaM; Mixtral 8x7B; DeepSeekMoE): the dominant production convention for learned-routing MoE. Two experts per token blend complementary knowledge and double activated expert capacity at the same expert count.

**Justification for 2.** The top-1 case is a *communication-and-complexity* argument made at trillion-parameter scale with hundreds of experts per layer — that regime is not this one. Here the model is ~1B dense-equivalent and the per-step all-to-all volume is modest; with 16 experts a single expert per token removes the blending that top-2 is valued for and, at the same capacity, measurably loses quality (the skill's own risk note: "Top-1 quality loss vs Top-2"). Because this is a training-quality-focused run, top-2 is the default; the interconnect concern does not override it because absolute dispatch volume at this scale (≈ top-2 × tokens × 3 KB/token dispatch) is small next to compute.

**Revise to top-1 only if:**
- Profiling shows all-to-all/bubble consuming >~25–30% of step time (the communication bottleneck check), or
- Inference latency becomes the priority (top-1 halves dispatch), or
- Router entropy is persistently high enough that the second expert adds little.

### 2. Capacity factor = 1.25

**Conflicting positions weighed.**
- **1.0** — Switch's own speed ablation: Switch-Base at cf 1.0 was fastest (1000 ex/s) at a small quality cost (NLP −1.561 vs −1.553 at 1.25). Maximally efficient, minimal padding, but drops tokens under any imbalance.
- **1.25** — GShard/GLaM convention; Switch-Base at 1.25 matched the best quality and beat cf 2.0 on time-to-quality (65.0 h vs 72.8 h).
- **2.0** — Switch's main-scaling choice (top-1, essentially no drops) at the cost of ~2× padding compute and 2× dispatch/all-to-all volume.
- **No-drop padding** — Mixtral and DeepSeek drop no tokens at all (pad to max load); effectively unbounded capacity, only affordable when memory and interconnect are generous.

**Justification for 1.25.** Memory is not binding here (30 GB/GPU vs 80 GB), so the memory-pressure argument that pushes large-scale systems to 1.0 does not apply. The router is top-2, which roughly doubles per-expert load variance relative to Switch's top-1, so cf 1.0 would drop tokens under ordinary imbalance — dropped tokens starve under-loaded experts of gradients and cost quality. cf 2.0 is rejected: at this scale with a healthy aux loss, overflow is expected <1%, and 2.0 would double padding compute and all-to-all traffic on exactly the resource (standard interconnect) we are told is constrained. 1.25 absorbs the ~25% routing slack that top-2 + a 0.01 aux loss leaves, holding drops below ~1% for near-optimal throughput.

**Revise:**
- **Down to 1.0** if measured overflow/drop fraction is ~0 over many steps and router entropy stays high (pure throughput win).
- **Up to 1.5** if the drop fraction exceeds ~1–2%, or during the first ~2k steps before the router balances.

### 3. Auxiliary-loss coefficient = 0.01

**Conflicting positions weighed.**
- **0.001** — HF defaults (Mixtral, Switch, DeepSeek-V2 config) and DeepSeekMoE 16B. Very safe against distortion, but weak; the skill flags an absent/too-low aux loss as the #1 cause of router collapse.
- **0.01** — Switch Transformer (explicitly swept 10⁻¹→10⁻⁵, chose 10⁻² as "balanced load quickly without interfering with training loss"); DeepSeekMoE 2B; the Megatron-LM Mixtral example (`--moe-aux-loss-coeff 1e-2`).
- **0.02** — the actually-released Mixtral-8x7B config, paired with a router z-loss.
- **0.1** — top of the reported range; distorts the routing objective and forces near-uniform routing, killing expert specialization.

**Justification for 0.01.** The load-balancing loss is doing real work in this design: it must keep per-expert load inside the 1.25 capacity buffer (else drops) and inside the fixed 8-rank expert partition (else idle GPUs under EP=8). A coefficient of 0.01 is the most-tested value in the literature, sits one order of magnitude above the weak/default HF setting that risks collapse on a small model, and one order below the distortion threshold. I considered 0.02 (Mixtral's released value) but chose 0.01 because it is better-documented and the two are statistically indistinguishable; 0.01 keeps enough routing freedom for specialization while still guaranteeing load balance.

**Revise:**
- **Up to 0.02–0.05** if effective expert count < ~50% of total (collapse) or overflow climbs — measured via the router-distribution analyzer.
- **Down to 0.001–0.003** if the router becomes too uniform (entropy near max, no specialization) and quality trails the dense baseline, or if a short fixed-seed ablation confirms stability at the lower value (the DeepSeek 16B result shows the small coefficient can be viable when load balance is otherwise cheap).

---

### Cross-parameter consistency

The three choices reinforce each other: aux 0.01 keeps loads balanced enough that cf 1.25 rarely drops tokens; cf 1.25 absorbs the residual imbalance aux 0.01 doesn't remove (avoiding the top-2 dropout and gradient-starvation failure mode); top-2 gives the quality the run is for, and its extra dispatch cost is affordable precisely because 1.25 (not 2.0) bounds all-to-all volume on the standard interconnect. Watch three signals to keep the triple honest: **overflow/drop fraction** (validates cf), **effective-expert count / entropy** (validates aux), and **all-to-all share of step time** (validates top-k).

---

Want me to also draft the training config and launch plan (DP/EP layout, micro-batch) that this decision feeds into?