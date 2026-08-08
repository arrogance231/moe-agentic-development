# MoE Routing Decision Document — 1B Dense-Equivalent on 8× 80GB (Standard Interconnect)

**Date:** 2026-08-08 · **Budget:** 8 × 80GB HBM (640GB total), standard (non-NVLink-full-mesh) interconnect · **Target:** 1B dense-equivalent active parameters

## 1. Decision summary

| Parameter | Chosen value | Rejected range |
|---|---|---|
| Top-k routing | **k = 2** | 1 |
| Capacity factor | **CF = 1.25** | 1.0, 2.0 |
| Auxiliary-loss coefficient | **α = 0.01** | 0.001, 0.1 |

This is the GShard/ST-MoE/DeepSeek production triple: **top-2 + CF 1.25 + α 1e-2**.

## 2. What the budget actually stresses

Before weighing the knobs, note what the constraints do and don't bind:

- **Memory is not the binding constraint.** A 1B dense-equivalent MoE (e.g., 8 experts, active 2) has ~6–8B total parameters. Full training state (bf16 weights + fp32 master + Adam m/v + grads ≈ 16–18 B/param) is ~120–140GB — well inside 640GB, leaving room for activations and comm buffers. This is why we can afford padding (capacity headroom) and a second active expert: both are FLOPs/comm costs, not memory costs.
- **Communication is the stressed resource.** All-to-all dispatch/combine volume scales linearly with **top-k** and with **capacity factor** (padding tokens are still shuffled). Expert placement should be 1 expert/GPU across 8 GPUs, with non-expert layers replicated (GShard's "one expert per core" rule), so the only cross-GPU traffic is token dispatch — which makes the k and CF choices directly visible in the comm profile.

## 3. Top-k routing: k = 2

**Conflicting positions weighed.**
- **Top-1 (Switch Transformer)** — minimal dispatch traffic (half of top-2), simplest capacity bookkeeping, scales FLOPs savings maximally. It was the right call for trillion-parameter runs where per-token FLOPs are the whole game.
- **Top-2 (GShard, ST-MoE, Mixtral, DeepSeek-MoE)** — two experts per token average out router error, add representational capacity per token, and smooth the load distribution, which is what makes a tight capacity factor viable. Mixtral specifically re-adopted top-2 in decoder-only production after Switch argued for top-1.
- **Expert Choice (Zhou et al. 2022)** — at *equal compute*, expert-choice beats top-1 gating even at CF 0.5–1.0; the paper reads as evidence that top-1's single-slot routing — not the capacity mechanism — is the weak link.

**Justification for k=2.** At 1B dense-equivalent the model is *small*: per-token expert capacity is the quality bottleneck, and the router error from a single expert slot is a larger fraction of total signal than it is at frontier scale. Top-2 buys quality-per-FLOP headroom at the only price that matters here — doubled dispatch traffic — and we can amortize it: pack long sequences to amortize fixed all-to-all latency, overlap dispatch with GEMM, and run large microbatches. At H≈2048–4096 and packed batches, measured all-to-all should land in the ~10–20% of step-time band (top-1 would halve this — see §7 for the trigger that makes top-1 the right answer instead).

## 4. Capacity factor: CF = 1.25

**Conflicting positions weighed.**
- **CF = 1.0** — Switch's minimal-memory setting; Mixtral/Megablocks and modern Megatron drop-free dispatch run at *effective* CF 1.0 with no padding. The catch: with top-2 and any router noise, CF 1.0 drops tokens, and each dropped token silently loses an expert's computation (residual pass-through).
- **CF = 1.25** — the GShard production value; 25% padding buffer absorbs routing variance, and is the commonly recommended default (Switch authors found 1.25 the sweet spot between memory and drop rate).
- **CF = 2.0** — generous buffer, near-zero drops, but doubles expert padding compute and dispatch volume for marginal quality; only justified for small batches / very noisy routers.

**Justification for CF = 1.25.** With α = 0.01 and top-2, steady-state imbalance typically sits at 5–10%, so the 25% headroom is *almost unused* — the effective padding cost is small, not 25%. But during warmup and router transients the headroom absorbs load spikes and keeps the drop rate ≈ 0, which protects the fragile small model. The trade arithmetic favors 1.25 here: a single drop costs a token's full expert pass (a hard quality loss), whereas padding 25% of expert FLOPs costs only spare FLOPs we already have. CF 2.0 wastes comm we don't have; CF 1.0 risks drops we can't afford.

## 5. Auxiliary-loss coefficient: α = 0.01

**Conflicting positions weighed.**
- **α = 0.001** — too weak for top-2 with a tight CF: residual imbalance of 10–30% pushes tokens into the capacity ceiling, converting the CF into the primary balancing mechanism (i.e., dropping). Interactive/empirical sweeps show α = 0.001 fails to prevent expert collapse.
- **α = 0.01** — the Switch default; empirically reliable across Switch, GShard, ST-MoE, and DeepSeek-MoE's device-level balance loss. Contributes ~0.5% of the gradient signal, so it balances without distorting router specialization.
- **α = 0.1** — aggressively forces uniformity; Switch observed quality degradation at larger coefficients, and sweeps show α ≥ 0.1 distorts the router's task-loss gradient and raises task loss.

**Justification for α = 0.01.** The governing rule is: *the aux loss must be strong enough that the capacity factor rarely has to drop*. α = 0.01 keeps imbalance inside the CF = 1.25 headroom (drops stay below ~1%) while leaving the router free to specialize. Top-2's natural smoothing means we could plausibly run α = 0.005 and still hold the headroom, but 0.01 is the safer default given the 1.25 ceiling we committed to in §4 — the two choices are coupled, not independent.

## 6. Why the triple is coherent

The three knobs interact; choosing them as a set matters more than any single value:

- **Top-2 lowers the α required** (two slots per token smooth load) **and makes CF 1.25 sufficient**.
- **α = 0.01 keeps imbalance under 25%**, so the CF 1.25 headroom is padding, not drops — and the padding is cheap because memory/FLOPs are abundant and comm is the only thing we're spending (25% padding on top of ~0.26× dense compute is negligible in absolute terms).
- **CF 1.25 caps worst-case imbalance at 25% even if α fails transiently** (warmup, data skew), giving a hard safety net under the soft loss.

## 7. Conditions for revision (measurable triggers)

| Trigger | Action |
|---|---|
| Token-drop rate > 1–2% (over a moving window) | Raise α → 0.02–0.05 *or* raise CF → 1.5; re-measure |
| Imbalance ratio (max/mean expert load) > 15% at α = 0.01 | Raise α to 0.02 |
| Imbalance < 3% sustained for ≥50k steps | Lower α → 0.005 to free the router; keep CF 1.25 |
| All-to-all/dispatch time > 20% of step time (comm-bound) | Switch to **top-1** (Switch recipe: CF 1.0–1.25, α = 0.01), or adopt drop-free dispatch (Megablocks-style dense masks) to remove padding tokens from the wire, or use a flex/DeepEP dispatcher with all-gather instead of all-to-all |
| Quality trails a dense 1B baseline at equal steps | Add ST-MoE z-loss (logit magnitude stability); consider raising CF to 1.5 for capacity |
| Small batches / short sequences (routing variance ↑) | Raise CF → 1.5 (small batches average poorly, need more buffer) |
| Hardware upgrade to NVLink-full-mesh | Comm is no longer stressed: keep top-2 and tighten CF → 1.0 with drop-free dispatch |
| Target grows beyond ~1B dense-equivalent | Re-derive: memory and padding both bite; likely CF → 1.0 drop-free + higher α |

## 8. Monitoring checklist (what to log from step 1)

Per-MoE-layer: tokens dropped (%), per-expert load histogram, imbalance ratio (max/mean), aux-loss value, router logit magnitude (for z-loss need), and per-step dispatch/combine time share vs GEMM time. Decide revisions on these, not on validation loss alone — a collapsed router can look fine on task loss while under-utilizing the model.

## 9. References weighed

- Lepikhin et al., *GShard* (2020) — top-2, CF ∈ [1,2] (1.0 adequate at 2048 experts; 1.25 production norm).
- Fedus et al., *Switch Transformer* (2021) — top-1, CF ~1.25 default, α = 0.01 default.
- Zoph et al., *ST-MoE* (2022) — top-2, CF 1.0, introduces router z-loss.
- Jiang et al., *Mixtral of Experts* (2024) — top-2, drop-free dispatch, no aux loss.
- Zhou et al., *Expert Choice Routing* (2022) — EC beats top-1 gating even at CF 0.5–1.0 at equal compute.
- Dai et al., *DeepSeekMoE* (2024) — fine-grained top-2+shared experts, device-level balance loss α = 0.01.

---

*(Deliverable: this document as `MoE_hyperparameter_decision.md` under `benchmarks/harness/configs/A0/`.)*