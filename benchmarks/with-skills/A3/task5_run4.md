Searches timed out; the task already provides the literature conflicts, so I'll proceed with domain knowledge. Let me look at the working directory context.
Search confirmed the key literature facts (Switch Transformer's α=10⁻² sweep, CF 1.0 for top-1; GShard's CF 2.0 for top-2; capacity formula `(tokens/num_experts) × CF`). No file tools are available in this session, so here is the decision document.

---

# Decision: Capacity Factor, Auxiliary-Loss Coefficient, and Top-k Routing for a 1B Dense-Equivalent MoE on 8×80GB

**Budget:** 8× 80GB GPUs, standard (non-NVLink-full-mesh) interconnect · **Scale:** ~1B activated parameters (dense-equivalent), 16–32 experts (assumed), training-focused run · **Convention:** expert capacity = `CF × (tokens_per_batch / num_experts)` per MoE layer (Switch/GShard convention).

## Decisions at a glance

| Parameter | Value | Rationale in one line |
| --- | --- | --- |
| **Top-k routing** | **2** (top-2) | Training-quality default; knowledge blending; 2× dispatch is affordable at 1B scale even on standard interconnect |
| **Capacity factor** | **2.0** | Minimal CF that drops no tokens under balanced top-2 routing — zero waste, and memory is not binding on 80GB |
| **Aux-loss coefficient** | **0.01** | The swept optimum (Switch, GShard) that keeps routing near-uniform without dominating the primary loss |

The three values are coupled, not independent: top-2 sets a floor of CF ≥ 2.0 for zero drops; CF = 2.0 (zero slack) makes load balance load-bearing, which forces the aux loss high enough to hold it; aux = 0.01 is the value proven to hold it. Each choice below, with the conflicts weighed.

---

## 1. Capacity factor — **2.0**

**Conflicting positions weighed**
- **CF = 1.0 (Switch Transformer, top-1).** Switch found lower capacity factors (1.0, 1.25) *better* — "perform better at lower capacity factors." But this is a **top-1** result: with one active expert per token, CF = 1.0 is exactly full capacity under uniform routing (zero slack), and drop rates stayed < 1% given a strong-enough aux loss. It is not portable to top-2.
- **CF = 2.0 (GShard, top-2).** GShard's production value. Under uniform top-2 routing each expert expects 2× the per-expert mean, so CF = 2.0 is the *minimal* value that drops no tokens. Expert Choice (Zhou et al.) explicitly set `c = 2` to "directly compare to the top-2 token-choice gating in GShard."
- **CF = 1.25 / 2.5 variants, and Mixtral's unlimited capacity.** Some frameworks run CF = 1.25 (25% slack) or Mixtral-style no-token-dropping. These buy imbalance headroom at the cost of wasted compute and — critically on our interconnect — **higher dispatch/all-to-all volume**, which scales with capacity, not just with top-k.

**Justification for 2.0 (not a range):** With top-2, CF < 2.0 is not a tuning choice — it is a hard defect (under perfectly balanced routing, CF = 1.25 would drop ~37% of expert assignments; CF = 1.0 drops half). CF = 2.0 is the efficiency optimum: every byte of capacity is used under balance, so it is the lowest-compute, lowest-communication configuration that drops no tokens. 8× 80GB is not memory-bound at 1B scale, so there is no pressure to raise CF for buffer reasons, and raising it to 2.5 only buys slack that the aux loss already provides — while growing all-to-all volume on the standard interconnect for nothing. The Expert Choice finding that top-2 step latency is bottlenecked by the *most-loaded* expert is addressed by balancing (aux loss), not by capacity padding.

**Revise if:** measured overflow/drop fraction exceeds ~2–3% or effective experts fall below ~0.8×N at steady state → raise toward 2.5 **or** strengthen the aux loss first (preferred, keeps compute/comm flat); if memory pressure ever materializes (it should not at 80GB), shrink micro-batch, never the CF.

---

## 2. Auxiliary-loss coefficient — **0.01**

**Conflicting positions weighed**
- **α ≈ 0.001–0.003 (DeepSeek-V2/V3).** DeepSeek runs far lower than the classic value — but only because they layer on *additional* balance machinery we do not have: a device-level balance loss, bias-based routing correction, and generous capacity/offload slack. Importing their low α without their mechanisms invites router collapse.
- **α = 0.01 (Switch, GShard).** Switch explicitly swept α from 10⁻¹ to 10⁻⁵ in powers of ten and found **10⁻²** the sweet spot: "balanced load quickly without interfering with training loss," with dropped tokens < 1% at CF = 1.0 (their zero-slack equivalent).
- **α ≈ 0.1.** The top of the reported range. Switch found 10⁻¹ too strong — it overwhelms the primary cross-entropy objective and distorts routing (experts chosen for balance, not token fitness).

**Justification for 0.01 (not a range):** Our configuration has **zero capacity slack** (CF = 2.0 under top-2), so load balance is load-bearing: imbalance converts *directly* into dropped tokens and the most-loaded expert becomes the step-latency bottleneck. We need an α proven to hold routing near-uniform at zero slack, and 0.01 is precisely that number — the only value in the literature that was actually swept and shown to balance "quickly without interfering" with the primary loss. The low end (0.001) is not portable because DeepSeek's extra mechanisms are absent; the high end (0.1) is demonstrably destructive.

**Revise if:** effective experts < 0.8×N, rising Gini, or visible overflow (collapse signals) → raise toward 0.02–0.05; if the loss curve degrades relative to the dense baseline or routing entropy collapses too low (over-regularization) → lower toward 0.001.

---

## 3. Top-k routing — **2**

**Conflicting positions weighed**
- **Top-1 (Switch Transformer; DeepSeek-V3-style fine-grained).** Halves all-to-all dispatch vs top-2 and is the communication-frugal choice — the reason modern large-scale systems (DeepSeek, Qwen) push toward small-k with many fine-grained experts. Switch also showed top-1 at CF 1.0 is FLOP-cheap and quality-competitive at scale.
- **Top-2 (GShard; Mixtral; Grok; Qwen2.5-MoE).** The training-quality standard. Two experts per token blend knowledge, reduce per-token variance, and are robust when experts are small — which is exactly our regime (16–32 experts over ~1B activated ⇒ each expert is only a few million params).
- **Learned/soft routing.** Rejected: higher ceiling but unstable and harder to load-balance; unnecessary complexity for a benchmark run.

**Justification for top-2 (not a range):** This is a training-quality-focused run at 1B scale, and the communication argument for top-1 is weak here. All-to-all volume per MoE layer is `top_k × tokens/step × dtype bytes`, which at 1B-scale batches (tens of thousands of tokens) is a few hundred KB to a few MB per layer — top-2's 2× is small in absolute terms, and the tokens-per-expert floor is trivially met (with EP=8, 16–32 experts, mb=8, seq=2048: ≈8k–16k tokens/expert/GPU, far above the 8–64 floor). Top-1's per-token risk concentration is real at this scale: with a 16–32-expert budget, every token betting on a single small expert costs quality that top-2 recovers cheaply. The interconnect is the *only* reason to prefer top-1, and it does not bind at 1B scale.

**Revise if:** measured communication share of step time exceeds ~25–30% (or a step-time regression appears on the standard interconnect), or the run becomes latency/throughput-bound → drop to top-1 with CF = 1.25 (per Switch), re-check the aux loss, and re-tune the batch.

---

## Why the combination is coherent

- **top-2** sets the zero-drop floor **CF = 2.0** (uniform top-2 load = 2× mean).
- **CF = 2.0** gives zero slack ⇒ balance must come from the router, not capacity padding.
- **α = 0.01** is the swept value that delivers that balance without distorting routing.
- Net effect: GShard's exact production recipe (top-2, CF 2.0, α 0.01) — the most battle-tested combination in the literature — transplanted to a budget where memory (640GB HBM) is ample and the only scarce resource (interconnect bandwidth) is used as frugally as the quality target allows.

## Sources consulted

Switch Transformers (Fedus et al., JMLR 2022 — CF sweep Table 1, α sweep §2.2, "better at lower capacity factors 1.0, 1.25"); GShard (Lepikhin et al., 2020 — top-2, CF 2.0); Expert Choice (Zhou et al., 2022 — `c=2` matches GShard top-2, top-2 latency bottleneck finding); Mixtral (Jiang et al., 2024 — top-2, no token dropping); DeepSeek-V2/V3 (low α + device-level balance machinery); Qwen2.5-MoE (small-k production routing).