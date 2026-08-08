Note: no file-write tool is available in this session, so the decision document is delivered inline below.

```markdown
# MoE Hyperparameter Decision — 8×80GB, 1B Dense-Equivalent

## Decision summary

| Parameter            | Chosen value |
|----------------------|--------------|
| Top-k routing        | **k = 1** (top-1) |
| Capacity factor      | **1.25** (normalized to per-expert mean load ⌈S/E⌉) |
| Auxiliary-loss coefficient (α) | **0.01** |

These three values form a matched, "efficiency-tuned, balance-enforced" set. They
are chosen so that the two scarce resources on this budget — all-to-all dispatch
bandwidth on a standard (non-full-mesh) interconnect, and expert utilization —
are optimized, while memory (which is abundant here) is treated as non-binding.

## 0. Budget analysis (what actually constrains us)

- **Memory is not the constraint.** A 1B dense-equivalent MoE with 8 experts and
  top-1 routing carries roughly 3–6B total parameters (top-1 maximizes the
  total-parameter-per-compute ratio: `T = a + E·p`, `A = a + k·p`, so with `E=8`
  and `A=1B`, top-1 → `T≈5.9B`, top-2 → `T≈3.1B`). Mixed-precision training with
  Adam occupies ~16 B/param → **50–100 GB total**, comfortably inside 640 GB
  HBM even after activations, comm buffers, and KV cache. We are not forced into
  small expert counts or low capacity factors by memory.
- **The interconnect is the constraint.** With expert parallelism, every routing
  decision moves tokens over all-to-all. Dispatched volume scales with
  `S × k` (token–expert pairs) and with the padding implied by the capacity
  buffer. On a standard, non-full-mesh link this bandwidth is the first thing to
  saturate, and its cost is measured in wall-clock, not just bytes. All three
  decisions below are driven by minimizing (a) `k`, (b) padding at fixed `k`,
  and (c) the imbalance variance that forces (a)/(b) upward.

## 1. Top-k routing → k = 1

**Positions weighed**

- **Top-1 (Switch Transformer, ST-MoE; most small-scale MoE).** One expert per
  token halves dispatch traffic versus top-2, simplifies load accounting (each
  token contributes exactly one count to an expert's load), and Switch's
  dense-to-sparse ablations found top-1 matches top-2 quality when an auxiliary
  loss and modest capacity slack are added. Recommended as the default for
  0.2B–2B dense-equivalent regimes.
- **Top-2 (GShard, GLaM, Mixtral, DeepSeek-V2/V3).** Higher quality per routing
  decision (a token draws on a combination of two experts), better robustness to
  a single overloaded expert, and a larger "capacity gain" at the same total
  parameter count. GShard pairs top-2 with an implicit capacity factor of 2.0.
  DeepSeek only makes top-2 affordable by adding *device/node-limited routing*
  to cap communication — an explicit admission that top-2's dispatch cost is a
  systems problem on bandwidth-constrained interconnects.

**Justification.** At 1B dense-equivalent with only 8 experts, each expert is
~90M FFN parameters — specialization granularity is coarse, so the marginal
quality from a second expert per token is small relative to the 2× all-to-all
volume it costs on this interconnect. Top-1 also gives a strictly better
parameter/compute ratio (≈5.9B total at 1B compute vs ≈3.1B for top-2), which is
exactly the spare capacity this memory-rich, bandwidth-poor node can exploit.
We do not need DeepSeek-style device-limiting machinery to make top-2 viable
because top-1 does not need it at all.

**Revise if:** (1) interconnect is upgraded to full-mesh NVLink or a
transformer-engine-class topology, making the 2× dispatch cost cheap; (2) held-out
evals show a persistent quality gap versus the dense 1B baseline that top-1 MoE
cannot close at acceptable steps; or (3) expert count rises substantially (top-2's
advantage grows with finer-grained experts, e.g. 32+ routed experts), in which case
revisit top-2 *with* device-limited routing (see §4).

## 2. Capacity factor → 1.25 (normalized)

**Convention note (this is why the literature looks contradictory).** The reported
range 1.0–2.0 mixes two conventions:

- GShard-style: `capacity = CF · ⌈S/E⌉` per dispatch, where top-2 needs CF ≈ 2.0
  just to absorb average load ("the implicit capacity factor in GShard is exactly
  2 — just enough to accommodate top-2 routing under uniform load").
- Switch-style: CF is applied to the *mean per-expert load for the chosen k*, so
  top-1 with CF 1.0 and top-2 with CF 2.0 are the *same* operating point.

All values below use the Switch (normalized) convention: `CF = 1.0` means zero
slack over the average expert load under the chosen k.

**Positions weighed**

- **CF 1.0** (Switch, ST-MoE). Zero padding; relies on the auxiliary loss to hold
  balance. Brittle early in training, when routers are uncalibrated; expert-choice
  routing observed token-choice over-capacity ratios of 20–40% for some experts
  even with aux-loss control, i.e. real dropping at CF 1.0.
- **CF 1.25** (GLaM; the de-facto 2026 production default per Megatron-Core /
  DeepSpeed MoE guidance; "recommended default of 1.25... increase to 1.5 or 2.0
  only on high drop rates"). Small slack absorbs residual imbalance at a bounded
  padding cost (≤ ~20% of dispatch slots).
- **CF 1.5–2.0** (GShard's implicit 2.0 under its own convention; safety-first
  production configs; expert-choice's CF2 baseline). Near-dropping-free but wastes
  up to ~50% of dispatch slots on padding, and padding bytes are still *sent* over
  the interconnect in standard fixed-buffer all-to-all implementations.

**Justification.** With k=1, CF 1.0 is zero-slack and drops real tokens on a
normal interconnect; CF 2.0 spends half the dispatch bandwidth on padding. CF 1.25
sits just above the dropping floor: it keeps padding at a manageable ~20% of
dispatch slots while covering the load variance a well-trained top-1 router with
aux loss actually produces. It is the empirically dominant value in the literature
that spans both regimes, and it costs us little here because dispatch bytes at
1B-scale are small in absolute terms — the 1.0→1.25 step buys robustness for a
modest slice of interconnect budget, whereas 1.25→2.0 buys almost nothing but
padding.

**Revise if:** (1) measured token drop rate exceeds ~1% of tokens → raise to 1.5
(do not wait; dropping is silent quality loss); (2) padding consistently exceeds
~10% of dispatch slots over several checkpoints *and* balance CV is well-controlled
→ tighten toward 1.0 and compensate with a slightly higher aux coefficient; or
(3) batch size shrinks (e.g. short-sequence fine-tuning), which raises per-batch
load variance → increase CF for that run.

## 3. Auxiliary-loss coefficient → 0.01

**Positions weighed**

- **α ≈ 0.001** (low end). Minimal interference with the language-modeling
  objective; used where balance comes mostly from elsewhere — e.g. Mixtral's
  router z-loss only, or DeepSeek-V3's auxiliary-loss-free *bias-based* balancing,
  whose companion sequence-wise loss uses an "extremely small" α. In the
  Loss-Free Balancing paper the best softmax-gate α tuned down to ≈0.0003.
- **α = 0.01** (Switch Transformer's default; ST-MoE; DeepSeek-V2). The empirical
  sweet spot: strong enough to hold expert load near-uniform at tight capacity
  factors, weak enough not to visibly regress perplexity. Switch's ablations
  report this as reliable across expert counts.
- **α = 0.1** (high end). Near-forced uniformity; reliably degrades routing
  specialization and downstream quality (Switch, expert-choice, and
  DeepSeek-V3/Loss-Free papers all document the "large α hurts performance" trade).

**Justification.** We run at a tight CF (1.25), so balance must be actively
enforced — that rules out the 0.001 end, where imbalance would appear as either
dropping (worse quality) or as forced slack (worse interconnect efficiency).
0.1 rules itself out by degrading the router's specialization. 0.01 is the value
that the largest body of production evidence (Switch, ST-MoE, DeepSeek-V2)
converges on for exactly this operating point (top-1, CF≈1.0–1.25, expert
parallelism), and it is the coefficient that keeps per-expert load CV near the
~1.5% level MLPerf's DeepSeek-V3 benchmark treats as healthy.

**Revise if:** (1) drop rate > ~0.5% despite CF 1.25 → raise α toward 0.03 before
touching CF, since dropping is cheaper to fix at the balance source; (2) eval
perplexity or downstream metrics regress while balance is good → step α down
toward 0.003; or (3) we adopt DeepSeek-V3-style bias-based (aux-loss-free)
balancing or a z-loss router → drop α to ~0.0003–0.001 and rely on the bias
mechanism / z-loss instead, keeping a small sequence-wise α as a safety net.

## 4. Why these three are a matched set (and the adjacent configs)

The three parameters are not independent. Raising k raises mean per-expert load,
which raises dropping pressure and therefore pushes CF up or α up. Lowering α
requires more CF slack (or bias balancing) to avoid dropping. The chosen tuple
(k=1, CF=1.25, α=0.01) is the coherent point that minimizes interconnect volume
while holding dropping ≈ 0. Coherent alternatives, to be revisited under the
trigger conditions above:

| Configuration | When it applies |
|---|---|
| k=1, CF=1.25, α=0.01 | **chosen** — bandwidth-constrained, memory-rich, 1B-equiv, 8 experts |
| k=2, CF=1.25, α=0.01–0.03 | quality-first; only after interconnect upgrade or fine-grained experts |
| k=1, CF=1.0, α=0.03 | near-zero padding required; accepts slight drop risk, uses α to hold balance |
| k=1, CF=1.5, α=0.003 | interconnect is cheap relative to quality; prefers slack over aux interference |
| k=2 + device/node-limited routing, CF=1.25, α≈0.01 | DeepSeek-style; the way to get top-2 quality on a bandwidth-poor fabric |

## 5. Monitoring and revision triggers

| Metric | Target | Action if violated |
|---|---|---|
| Token drop rate | < 0.5% (alarm at 1%) | Raise CF to 1.5, then raise α to 0.03 |
| Padding fraction of dispatch slots | < 10% sustained | Lower CF toward 1.0, raise α to hold balance |
| Expert-load coefficient of variation (CV) | ≈ 1–2% | Raise α; if CV stays high, inspect router init / temperature |
| Eval perplexity / downstream delta vs dense 1B baseline | no regression at matched steps | Lower α to 0.003; if still regressing, reconsider k |
| All-to-all comm time share of step time | stable, < ~20% | If rising, re-check padding and k; consider device-limited routing |
| Router logit magnitude drift | stable | Add z-loss (coefficient ~0.001) rather than raising α |

**Bottom line:** k=1, capacity factor 1.25, auxiliary-loss coefficient 0.01 — the
efficient, balance-enforced operating point the literature converges on for
bandwidth-limited expert-parallel training, chosen because on this budget the
binding resource is interconnect dispatch, not HBM.
```

---

Key reasoning in brief: memory (640 GB) is non-binding even at 5.9B total params, so the decisions hinge on the interconnect. Top-1 halves all-to-all volume vs top-2 and maximizes params-per-compute at 1B dense-equiv; CF 1.25 sits just above the zero-slack floor where padding (~20% of dispatch slots) is a bounded, not dominant, interconnect cost; α=0.01 is the literature's convergent value for exactly this (top-1, CF≈1.25) operating point — the 0.001 end would drop tokens at tight capacity, and 0.1 demonstrably degrades routing specialization.