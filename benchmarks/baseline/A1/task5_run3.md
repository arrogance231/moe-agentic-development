# MoE Configuration Decision — 8×80GB, Standard Interconnect, 1B Dense-Equivalent

**Decision:** For an 8×80 GB GPU node with non-full-mesh interconnect, targeting a 1B dense-equivalent model with 8 experts (1 expert/GPU, expert-parallel all-to-all), I select:

| Parameter | Chosen value |
|---|---|
| Top-k routing | **2** |
| Capacity factor (CF) | **1.25** |
| Auxiliary-loss coefficient (α) | **0.01** |

Conventions used: *capacity = ⌈(k·T/E)·CF⌉*, i.e. the "fair share" already accounts for k=2. Under the Switch (top-1) convention where capacity = ⌈(T/E)·CF⌉, the equivalent setting is CF ≈ 1.25 with k doubled implicitly — see §4 for why this convention distinction matters.

---

## 1. Scenario and binding constraints

- **Memory is not binding.** 640 GB total VRAM holds a 1B-active model (≈2.5–4.8 B total params for 8 experts) many times over; expert buffers at any CF are <5% of per-GPU memory.
- **Compute is not binding.** Active FLOPs ≈ 6·10⁹·T per step; on 8×H100-class this is ~0.05–0.1 s/step.
- **The interconnect is the binding resource.** Every MoE layer requires all-to-all dispatch+combine proportional to `k · T · d`. With a standard fabric (PCIe/200–400 GbE/RoCE, no NVSwitch mesh), this is the first thing that shows up in a step-time profile and the primary reason published guidance on these three knobs disagrees.
- Deployment assumed: 8 experts, EP=8, all-to-all every MoE layer, ~16–24 MoE layers, batch tokens processed per layer-step per GPU in the thousands to low tens of thousands.

---

## 2. Top-k routing — **k = 2**

### Conflicting positions weighed
- **Top-1 (Switch Transformer, Fedus et al. 2021):** routing to a single expert roughly halves router arithmetic *and all-to-all communication volume* — the explicit motivation for the paper, since top-1 is "not strictly better per FLOP; it's cheaper." Enabled 1.6T-parameter scaling and is used where dispatch dominates (huge expert counts, e.g., 2,048).
- **Top-2 (GShard, GLaM, Mixtral):** production convention for MoE LLMs; Mixtral-8×7B (47B total, 13B active) and GLaM both use top-2 and show quality/capacity wins at matched active FLOPs. Key property: with 8 experts, k=2 gives **2× total parameters (capacity) for the same active-parameter compute budget** — exactly the MoE value proposition. Doubles dispatch volume but also doubles total capacity per byte of communication.

### Justification for k=2
- At 1B active, all-to-all per layer is on the order of `2·k·d·2 bytes` per token — for typical `d≈2048–4096` and moderate batch, this is **hundreds of MB per layer, low-tens of GB per step**, i.e. ~0.1–0.3 s/step on a standard fabric. That is overlappable with (and comparable to) the 0.05–0.1 s compute; it is a headwind, not a wall.
- The 2× capacity advantage (2.4–4.8B total params at identical 1B active) is the dominant lever for final quality at this scale, and per-communication-byte top-2 buys 2× the expert capacity.
- Top-1's communication win only becomes decisive when dispatch dominates step time — the regime Switch targeted (trillions of params, thousands of experts). We are two orders of magnitude below that.

### Revision triggers
- Downgrade to **top-1** if profiling shows all-to-all (dispatch+combine) > ~30% of step time, or if we need to fit the step in half the comm budget for throughput/latency-critical inference.
- Upgrade to **top-k>2** only if perplexity/evals lag the 1B dense baseline and measured comm headroom exists; with only 8 experts, k>2 marginalizes the gain (fewer distinct expert subsets per token).

---

## 3. Capacity factor — **CF = 1.25**

### Conflicting positions weighed
- **CF = 1.0 (tight):** DeepSeek-V2's device-level budget uses CF≈1.0 with lowest-affinity token dropping; Megablocks/Mixtral effectively run dropless/near-1.0 via block-sparse kernels. Cheapest — no padding waste — but drops tokens whenever routing is imperfectly uniform, which is always, especially early in training.
- **CF = 1.25:** Switch's recommended default; the paper's drop-rate vs. CF curve shows steep improvement up to ~1.25 and **diminishing returns beyond ~1.5**; GLaM also uses 1.25. With a well-tuned balancing loss, measured drop rates stay <1%.
- **CF = 2.0 (loose):** the GShard upper bound and the traditional non-top-1 MoE setting. Guarantees near-zero drops but up to 2× padding compute and **2× per-expert buffer → 2× dispatch/compute inflation**. On a comm-constrained fabric this is the worst option; Switch found 1.0–1.25 *better than 2.0* once balancing works.

### Justification for CF = 1.25
- CF multiplies expert-buffer size, padding compute, *and* padded-dispatch volume — everything we are short on (comm and compute efficiency) while buying nothing we are short on (memory). So we take the **smallest CF that keeps token-drop rate < ~1% given the α=0.01 balancing loss**.
- CF=1.0 is risky: with only 8 experts and a 1B model, per-expert load noise is proportionally larger early in training, so a strict buffer drops a few % of tokens before the aux loss flattens routing — a real quality tax that CF=1.25 removes cheaply.
- CF=2.0 is unjustified waste: it doubles the very resources the interconnect starves. 1.25 is the empirically tuned sweet spot (Switch, GLaM) for exactly this trade.

### Revision triggers
- Raise to **1.5 → 2.0** if measured per-layer drop rate exceeds ~1–2% after warmup (or `max_load/mean_load` stays > ~1.5), indicating the balancing loss is losing to data skew.
- Lower toward **1.0 (or dropless block-sparse, e.g., Megablocks-style kernels)** if drop rate stays <0.5% and padding/dispatch waste shows up as the top step-time line; with a dropless kernel you get CF→1.0 behavior without the drop tax.

---

## 4. Auxiliary-loss coefficient — **α = 0.01**

### Conflicting positions weighed
- **α ≈ 0.001 (weak):** the HF default for Mixtral and DeepSeek-V2's expert-level factor (α₁=0.001) — minimal interference with the primary loss; DeepSeek-V2 uses it because its fine-grained expert segmentation and device-level limits do most of the balancing work.
- **α = 0.01:** Switch **swept α from 10⁻¹ to 10⁻⁵ in powers of ten** and found 10⁻² "balanced load quickly without interfering with training loss"; this is the canonical tuned value (GShard's floor is also ~0.01).
- **α ≈ 0.1 (strong):** GShard's upper bound. Guarantees near-uniform routing but actively degrades quality by destroying expert specialization — the failure mode the literature consistently warns about.

### Justification for α = 0.01
- With only **8 experts and a small 1B model**, the risk calculus is asymmetric: (i) collapse risk is real because a dead expert is 12.5% of capacity, so the weak-coefficient regime (0.001) is too risky at this expert count — it relies on the fine-grained/decentralized balancing that 8-expert top-2 does not have; (ii) over-regularization (0.1) costs specialization that we need to extract value from the sparse model at all.
- α=0.01 is the middle of the two-orders-of-magnitude range, is the value the original authors actually swept and tuned, and it is **co-designed with CF=1.25**: enough balancing pressure to make a 1.25 buffer deliver <1% drops, weak enough not to flatten expert specialization.
- Mixtral's own released config sits at 0.02 (with z-loss 0.001) — the same order of magnitude; 0.01 is a defensible, slightly more conservative anchor.

### Revision triggers
- Raise to **0.02–0.1** if routing collapses (expert load coefficient-of-variation > ~0.1 across layers, dead experts, or drop rate climbing) — the standard fix is α: 0.001 → 0.01 → 0.02.
- Lower toward **0.001–0.005** if downstream metrics indicate over-regularization (experts too uniform, quality stagnates despite good balance), or if we adopt a bias-based/aux-loss-free balancing scheme (DeepSeek-V3 style), in which case this coefficient is retired entirely.

---

## 5. Monitoring plan that gates the revisions

Per MoE layer, log: token-drop rate, expert load CV / max-to-mean ratio, and wall-clock split of *all-to-all* vs *expert FLOPs* vs *attention*. Gate every 1k steps during warmup:

- Drop rate >1% ⇒ raise CF; drop rate <0.5% and padding-heavy profile ⇒ lower CF / go dropless.
- Load CV >0.1 or dead experts ⇒ raise α; balance "too perfect" (routing entropy at ceiling) with flat quality ⇒ lower α.
- All-to-all >30% of step time ⇒ drop to top-1, re-tune CF=1.25/α=0.01.
- Final quality gate: eval per-token against a 1B dense baseline; if MoE underperforms, prefer adjusting k and CF over α.

---

## 6. What would change this decision

- **Larger target model (e.g., ≥7B active) or more experts:** all-to-all volume grows linearly; at the point step time becomes comm-dominated, flip to top-1 + CF=1.25 (Switch recipe) or add device-limited routing (DeepSeek-V2) to cap cross-device traffic.
- **Full NVLink/NVSwitch mesh:** comm stops being the binding constraint; keep top-2, and you may raise CF toward 1.5 for quality without a perf tax.
- **Dropless kernels available (Megablocks/expert-choice):** retire the capacity factor entirely; keep k=2 and α=0.01, revisit only if dropped/over-processed token variance hurts inference-time quality.
- **Interconnect scale-down (e.g., 100 GbE):** comm becomes decisively dominant even at 1B; move to top-1 and consider trading k for CF — the single most defensible lever in that regime is exactly what Switch demonstrated: one expert per token, a 1.25 buffer, and a strong-enough balancing loss to keep the buffer mostly full.