I'll start by examining the working directory for context, then research the current literature on these MoE hyperparameters.
The skill isn't relevant to this task. Let me explore the working directory and research the literature in parallel.
Two more targeted searches to ground the DeepSeek/Mixtral positions and the interconnect angle.
## MoE Routing Decision for 1B Dense-Equivalent on 8×80GB GPUs

**Assumptions:** 8 experts placed one-per-GPU (expert parallelism = 8, so each layer's all-to-all spans all 8 devices), BF16 training, and — critical for interpreting every "capacity factor" below — the ST-MoE convention **capacity = ⌈CF × (k·T/N)⌉**, i.e. capacity measured in units of the *expected per-expert load* under top-k. In "top-1 units" (capacity = CF × T/N, the Switch/GShard usage) my chosen value reads as **CF ≈ 2.5**.

---

### Decision Summary

| Parameter | Value chosen |
|---|---|
| Top-k routing | **2** |
| Capacity factor | **1.25** (expected-load convention = 25% slack over perfect top-2 balance) |
| Auxiliary-loss coefficient | **0.01** |

---

### 1. What the budget actually constrains

640 GB of VRAM vs. a ~2–4 B-parameter MoE (~6–8 GB in BF16) makes memory a non-issue. The binding constraints are **compute and the interconnect**: with a 1B-active model the expert compute per token is small, so the expert-parallel all-to-all (dispatch + combine) is bandwidth/latency-bound, and its volume scales as `batch × seq × hidden × k × CF`. Non-full-mesh fabric therefore argues for *frugal-but-safe* k and CF, not maximal headroom. This framing drives every choice below.

---

### 2. Top-k: positions weighed

- **Top-1 (Switch Transformer; Clark et al. 2022).** Halves dispatch/combine traffic vs. top-2; Switch reached dense T5-11B quality at 7× speedup with a CF convention of 1.0–1.25. Minimal communication, but fragile: one expert per token, so any imbalance hits a hard capacity wall and drops tokens.
- **Top-2 (GShard, GLaM, ST-MoE, Mixtral).** Better quality per active FLOP; each token lands on two experts, which *naturally* spreads load (lower imbalance → fewer drops and a smaller needed aux-loss push); empirically robust to capacity tightening (Rectify-Router: cutting top-2 CF 2.0→1.5 costs only ~0.3 points). ST-MoE's systematic design sweep explicitly recommends top-2 with CF 1.25.

**Chosen: 2.** At 1B scale the per-token dispatch payload is small enough that 2× communication is affordable on this budget, while the robustness to imbalance and the load-spreading that lets me run a *lower* capacity factor (see §3) pay off directly in near-zero dropped tokens on a constrained fabric.

---

### 3. Capacity factor: positions weighed

The 1.0–2.0 literature range is partly an **artifact of convention**, not disagreement:
- **CF ≈ 1.0 (expected-load units).** Zero slack; any imbalance drops tokens. Equivalent to GShard's "CF = k" (Rectify-Router shows at CF=k, dropped = padding). Too tight for top-2 on real-world skewed text.
- **CF = 1.25.** 25% slack over expected top-2 load; the ST-MoE and GLaM production value; with an aux-loss-balanced router this gives ~0% drops at ≤25% padding overhead.
- **CF = 1.5–2.0.** GShard-scale and Mixtral-style no-drop buffers. Guarantee no dropping but up to 100% wasted expert slots and proportionally more all-to-all payload — the wrong direction for a communication-bound budget.

**Chosen: 1.25.** It is the highest value that does not waste the constrained resource (A2A bandwidth) and the lowest value that, together with top-2 + aux loss, still avoids the information-loss of dropped tokens. Padding is compute-only waste; dropped tokens are *quality* waste — 1.25 is the smallest buffer that trades only the former.

---

### 4. Auxiliary-loss coefficient: positions weighed

- **α ≈ 0.001 (DeepSeekMoE/V2 lineage; the aux-loss-free paper's baseline).** Keeps the router undistorted, but it belongs to a *dropless* design (top-1 + shared experts, device-level balancing, no hard capacity drops). In a capacity-limited top-2 router this is demonstrably too weak: the auxiliary-loss-free paper itself shows α = 1e-3 → 1e-2 → 1e-4 spans the collapse-to-over-regularization axis, and collapse here manifests as dropped tokens, which the capacity factor cannot absorb.
- **α = 0.01 (Switch default; GShard; Mixtral).** Contributes ~0.5% of gradient signal; enough to hold expert utilization near uniform (CV ≈ 1) without forcing the "fake uniformity" that GShard-era work warns about.
- **α = 0.1 (aggressive).** Guarantees balance but measurably degrades task loss (the large-α regime in the aux-loss-free analysis); only justified to rescue a collapsed router.

**Chosen: 0.01.** It is the only value in the surveyed range that keeps the balancing signal strong enough that the *capacity factor* does not have to do the balancing work by dropping tokens, while staying weak enough not to override router specialization. The 0.001 pole is invalidated precisely because I chose top-2 with a hard capacity limit; the 0.1 pole is invalidated because quality is the point of the exercise.

---

### 5. Why the triplet, not the individual optima

The three parameters are coupled. (top-2, CF 1.25, α 0.01) is the only surveyed combination that is simultaneously:
1. **near-zero token-drop** (top-2 spreads load; α=0.01 keeps it spread; CF=1.25 absorbs the residual skew),
2. **communication-frugal** (CF 1.25 caps padding/all-to-all payload at ~25%, vs. 2×–3× for GShard/Mixtral-style buffers),
3. **quality-non-destructive** (α=0.01, not 0.1, so routing stays task-driven), and
4. **directly validated in production** (ST-MoE 269B and GLaM both run top-2/1.25 with an α in this neighborhood).

Picking any single-parameter optimum breaks another: top-1 minimizes A2A but raises drop risk under skew; CF 2.0 kills drops but doubles the bandwidth that is this budget's scarcest resource; α 0.001 keeps the router clean but abdicates balancing to a capacity factor that is deliberately tight. I recommend adding ST-MoE's **router z-loss** (logit stabilization) as an orthogonal, coefficient-free safeguard.

---

### 6. When I would revise

| Trigger (measured, not guessed) | Revision |
|---|---|
| Profiling shows dispatch+combine > ~20–30% of MoE-layer time (fabric-bound) | Drop to **top-1** (halves A2A volume); keep CF 1.25 and α 0.01 in the new convention; accept the modest quality delta |
| Steady-state token-drop rate > ~1% despite balanced utilization (CV ≤ ~0.3) | Raise **CF to 1.5** first (cheap, communication still modest); raise α only if imbalance, not skew, is the cause |
| Task loss plateaus while balance loss keeps falling, or router probabilities go uniform | Lower **α to 0.003–0.005** |
| Data distribution shifts (e.g., code-heavy) and drops reappear | Bump CF, and re-check the α/drop interplay per layer (Skywork-style adaptive α is an option) |
| Fabric upgraded to full-mesh NVLink, or experts collapsed onto fewer GPUs (Megablocks-style, no cross-GPU A2A) | Relax: top-2/CF 2.0 (GShard) or more/finer experts become free; reconsider top-1's raison d'être |
| A dropless (Mixtral/DeepSeek-style) capacity scheme is adopted | Abandon CF entirely; revisit α toward the 0.001 pole, which is only valid in that regime |