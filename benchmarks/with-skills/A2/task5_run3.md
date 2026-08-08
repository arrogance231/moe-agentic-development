# MoE Hyperparameter Decision — 8×80GB GPUs, 1B Dense-Equivalent

## Decisions at a glance

| Parameter | Chosen value |
| --- | --- |
| Top-k routing | **2** |
| Capacity factor | **1.25** |
| Auxiliary-loss coefficient | **0.01** |

Context this choice is anchored to: **8× 80GB GPUs (640 GB HBM, ample for a 1B dense-equivalent MoE), non-NVLink-full-mesh interconnect (all-to-all is comparatively expensive), training objective** (not latency-bound inference).

---

## 1. Top-k routing = 2

### Conflicting positions weighed
- **Top-1 (Switch Transformer convention, CF ≈ 1.0, aux ≈ 0.01).** One expert per token: halves all-to-all dispatch volume, minimizes padding and per-token expert compute, and is the choice for inference/latency- or bandwidth-bound systems.
- **Top-2 (GShard / GLaM / Mixtral / DeepSeek-MoE convention).** Two experts per token roughly double effective expert capacity and blend expert knowledge; GShard showed top-2 substantially outperforms top-1 at the same total expert count, and it has been the standard for training-quality-focused work since.

### Justification for 2
The objective here is **training quality at 1B dense-equivalent**, not inference latency. Top-1's documented quality shortfall (halved active expert capacity, no knowledge blending, both noted as failure modes in the guidance) is not worth the savings. The interconnect concern is real but bounded: with 8 GPUs the expert-parallelism degree can be 8, so each all-to-all is a single collective over 8 ranks; top-2's 2× dispatch volume over that small topology is affordable relative to per-token expert FLOPs, and the 640 GB HBM budget means memory never binds us to the cheaper router. Quality is the binding constraint at this scale; take the quality default.

---

## 2. Capacity factor = 1.25

### Conflicting positions weighed
- **CF = 1.0 (Switch Transformer, top-1 pairing).** Maximally efficient, zero padding waste, but drops tokens the moment routing is imbalanced — starving under-loaded experts and silently shrinking the effective batch.
- **CF = 1.25 (GShard).** Absorbs ordinary token-routing imbalance with ~25% padding overhead; the most common production compromise.
- **CF = 1.5–2.0 (GLaM uses 2.0).** Effectively eliminates drops at high imbalance but pays 50–100% wasted expert compute in padding.

### Justification for 1.25
Early in training the router is unlearned and measurably imbalanced, so CF = 1.0 would drop tokens, distort the effective batch, and starve under-loaded experts (the collapse feedback loop in the guidance). GLaM's 2.0 wastes compute for no gain because the 0.01 aux loss keeps load balanced enough that 2.0's margin is unused. 1.25 sits at the point where typical imbalance is absorbed (per the 1.0–1.25 rule of thumb) and the waste is capped at 25%. On this hardware the padding cost is affordable: 640 GB of HBM means the capacity buffers fit trivially, and we can afford the extra expert compute in exchange for not dropping tokens.

---

## 3. Auxiliary-loss coefficient = 0.01

### Conflicting positions weighed
- **Low end ≈ 0.001.** Minimal distortion of the routing objective; later work (e.g. ST-MoE) found smaller coefficients can improve final quality relative to stronger ones **as long as collapse is still prevented**.
- **Canonical 0.01 (Switch, GShard, GLaM).** The most widely tested value; prevents expert imbalance/collapse across top-1 and top-2 settings.
- **High end ≈ 0.05–0.1.** Maximally aggressive load balancing, used mainly as a *corrective* lever once collapse is observed; at these strengths it visibly distorts the router's token–expert matching and typically hurts loss.

### Justification for 0.01
0.001 leaves too little margin at this scale: with top-2 and a fresh router the risk is drift into a collapsed/imbalanced regime with no strong signal to correct it. 0.1 distorts the routing objective and costs quality — it belongs to the fix-after-collapse toolkit, not the default. 0.01 is the well-tested default for exactly the pairing I've chosen (top-2 + CF 1.25): strong enough to keep the effective number of experts near n under CF 1.25, weak enough that the router still optimizes token-to-expert fit. Two orders of magnitude are spanned by literature; 0.01 is the best-supported interior point for a first training run.

---

## 4. Conditions under which I would revise

| Trigger | Evidence | Revision |
| --- | --- | --- |
| **Router collapse** | Effective experts < ~0.5·n via `router_distribution.py`; loss plateaus with few active experts | Raise aux loss 0.01 → 0.05 (then 0.1 if needed); verify CF ≥ 1.0 |
| **Excessive token drops** | Overflow fraction high in `router_distribution.py` despite balance | Raise capacity factor 1.25 → 1.5 |
| **All-to-all dominates step time** | Non-full-mesh interconnect; communication share of step time high | Prefer higher EP degree first; if still bound, drop CF to 1.0 and/or move to top-1 |
| **Quality shortfall vs dense baseline** | Loss/benchmark below dense-equivalent at matched compute | Lower aux loss toward 0.001 (ST-MoE finding) while monitoring effective experts; if gap persists, reconsider top-1 |
| **Precision/divergence issues** | NaN/spikes (`loss_analyzer.py`) | Fix precision/clipping first; only touch aux loss if routing distortion is implicated |
| **Deployment target shifts to inference** | Latency/throughput requirement appears | Switch to top-1 + CF 1.0, the conventional inference pairing |

**Revision principle:** change one lever at a time against a fixed seed, re-measuring effective experts, overflow fraction, and loss — never co-vary two of these three parameters when confirming a fix.

---

Note: I delivered this inline because this session has no file-write tools; if you want it written to `DECISION.md` in the config directory, say the word and I'll place it there.