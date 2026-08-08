I'll do a quick check on current MoE conventions (2025-2026 SOTA) to ground the routing/capacity-factor choices, then produce the doc.
# MoE-1B-DE: A 1B Dense-Equivalent Mixture-of-Experts Architecture

## Overview

`MoE-1B-DE` is a decoder-only transformer in which every FFN layer is replaced by a sparsely-gated Mixture-of-Experts (MoE). The design point is **"1B dense-equivalent"**: the per-token FLOPs and activated parameters match a ~1B dense transformer, while the *stored* parameter count is ~3.7B (4.1× sparsity ratio). We follow the Mixtral-style recipe (learned top-2 softmax gating) and use **fine-grained experts** (`expert_ffn_dim = d_model`), the same small-expert convention DeepSeek-V2/V3 popularized, which maximizes expert count for a given budget and improves load-balancing headroom.

The model is dense-equivalent, not dense-replicating: each token touches only the router, the two selected experts, and attention — never the full expert pool.

| Hyperparameter | Value |
|---|---|
| `d_model` (hidden) | 2048 |
| `num_layers` | 24 |
| `num_heads` / `head_dim` | 16 / 128 |
| `vocab_size` | 50,000 |
| `expert_ffn_dim` | 2048 (= `d_model`, fine-grained) |
| **num_experts `E`** | **16** |
| **active experts `top_k`** | **2** |
| routing | learned, softmax, deterministic top-2 |
| **capacity factor `CF`** | **1.25** |
| **auxiliary loss** | Switch-style load-balancing (`α = 0.01`) + ST-MoE router z-loss (`β = 0.001`) |
| embedding / output head | tied (counted once) |

## Parameters

Parameter math, per module. All values in millions (M).

**Embedding** (tied, counted once):
```
E_emb = vocab × d = 50,000 × 2,048 = 102.4M
```

**Per-layer attention** (Q, K, V, O — one 2048×2048 projection each):
```
E_attn = 4 · d² = 4 · 2,048² = 16.78M
```

**Per-layer router** (d × E gate logits):
```
E_router = d · E = 2,048 × 16 = 0.03M
```

**Per-layer experts** (each expert = gate/up 2048×2048 + down 2048×2048):
```
E_expert   = 2 · d · ffn_dim = 2 · 2,048 · 2,048 = 8.39M
E_experts  = E · E_expert = 16 · 8.39M = 134.22M
```

**Totals:**
```
Stored  = 102.4M + 24 · (16.78M + 0.03M + 134.22M) = 3,727M  ≈ 3.73B  (total)
Active  = 102.4M + 24 · (16.78M + 0.03M + 2·8.39M) = 908.5M  ≈ 0.91B  (per token)
Sparsity ratio = 3.73B / 0.91B ≈ 4.1×
Forward FLOPs/token ≈ 2 · 0.91B ≈ 1.8 GFLOPs  (≈ a 1B dense model's compute)
```

| Component | Params (M) | Per-token active (M) |
|---|---|---|
| Embedding (tied) | 102.4 | 102.4 |
| Attention ×24 | 402.7 | 402.7 |
| Router ×24 | 0.8 | 0.8 |
| Experts ×24 (16 per layer) | 3,221.3 | 402.7 (only 2 of 16/layer) |
| **Total** | **3,727 ≈ 3.73B** | **908.5 ≈ 0.91B (≈1B dense-eq.)** |

## Routing Choice

**Strategy: learned, deterministic top-2.** For each token, a single trainable gate matrix `W_r ∈ R^{d×E}` produces logits; softmax gives expert probabilities; the two highest-probability experts are selected and the token is split proportionally to those probabilities (Mixtral/GShard convention).

Justification against alternatives:
- **top-1 vs top-2**: top-1 (Switch, DeepSeek) is maximally sparse but puts every token on one expert — high variance, harder load-balancing, and no redundancy. top-2 costs only 2× the expert FLOPs while smoothing the optimization landscape and giving the load-balancer slack; it is the dominant choice in modern MoEs (Mixtral, Qwen3-MoE, DeepSeek).
- **learned vs fixed/random (hash) routing**: a learned gate adapts to data and is required for expert specialization; hash routing is cheaper but cannot specialize.
- **soft vs hard routing**: true soft routing (attend to *all* experts) eliminates sparsity and the FLOP savings; the whole point is *not* to touch the other 14 experts. Hence: soft *probabilities*, hard *selection*.

**Expert count & size (16 experts × 8.39M)**: fine-grained experts (DeepSeek-V2/V3 style) give more routing granularity per FLOP than a few large experts (e.g., Mixtral's 8×~4.4B FFNs), which both reduces per-expert gradient noise and raises `E`, making the load-balancing loss easier to satisfy.

**Capacity factor (CF = 1.25)**: each expert holds a fixed buffer of
```
capacity = ⌈(T · top_k · CF) / E⌉
```
tokens per batch, where `T` = tokens in the batch. The 1.25× headroom absorbs routing imbalance (tokens routed *above* capacity are dropped) with 25% memory waste. CF=1.0 (Mixtral) risks frequent drops; CF≥2.0 wastes memory and flattens the sparsity benefit. 1.25 is the standard sweet spot.

**Auxiliary loss**: two terms added to the LM loss.
1. **Load-balancing (Switch-style)**, coefficient `α = 0.01`:
   `L_bal = E · Σ_e f_e · P_e`, where `f_e` is the fraction of tokens routed to expert `e` and `P_e` the mean router probability for `e`. Minimizing `L_bal` drives uniform expert utilization. (`α` in the standard 0.01–0.1 range.)
2. **Router z-loss** (`β = 0.001`): `L_z = Σ log²(Σ exp(logits))` (ST-MoE), which discourages huge logits and eliminates the loss spikes z-loss-free balancing (DeepSeek-V3's bias trick) was designed around — simpler at this scale.

## Training Implications

- **Compute vs memory decoupling**: optimizer/activation memory is set by the **3.73B stored** params (BF16 + Adam states ≈ 29.6 GB/step for the full model), while throughput is set by the **0.91B active** — you get ~1B-model FLOPs but pay ~4× the dense memory. Intended; enables 1B-quality at 1B-compute cost.
- **Expert parallelism**: with 16 experts/layer, shard experts across GPUs (e.g., 2–4 experts/device) with data-parallel attention; tokens are dispatched via all-to-all, then reduced. At 3.7B this fits on a small node (e.g., 8×40GB), so a single all-to-all per MoE layer.
- **Per-expert batch is small**: average `E/2 = 8` experts handle each token, so each expert sees ~1/8 of the data-parallel batch. Mitigate with larger global batch size and a small expert-dropout.
- **Load-balancing is a training-time knob**: tune `α` (0.005–0.05) and watch the drop rate; if token-dropping exceeds ~0.5%, raise `α` or `CF`. Lower `α` late in training for better specialization (ST-MoE anneal).
- **Router stability**: z-loss plus standard warmup; router logits grow fast, so monitor z-loss and cap logits if needed.

## Risks

- **Load imbalance / token dropping**: the #1 failure mode — a few experts hoard tokens, others starve, and overflow tokens get dropped, degrading quality. Mitigation: `L_bal`, `CF=1.25`, per-expert dropout.
- **Router collapse**: the gate converges to always pick the same 2–3 experts, erasing the MoE benefit. Mitigation: z-loss, `α`, and monitoring `f_e` entropy.
- **Communication overhead**: all-to-all dispatch can dominate at small batch sizes; on multi-node runs, comms may exceed the compute savings if tokens/batch are too small.
- **Fine-tuning instability**: sparse models fine-tune less stably than dense (ST-MoE) — expect loss spikes; use lower LR, more warmup, and expert dropout during SFT.
- **Expert memory blowup**: capacity buffers scale with `E·CF`; 16 experts at CF 1.25 means ~1.4× raw expert storage in activations — must be factored into memory planning.
- **Serving complexity**: top-2 routing needs expert-parallel inference and KV/rank coordination; quantizing all 16 experts (vs 2 active) is required for practical serving.

---

**Summary of mandated specs**: `E = 16` experts, `top_k = 2` active, learned softmax top-2 routing, `capacity_factor = 1.25`, auxiliary load-balancing loss (`α = 0.01`) + z-loss (`β = 0.001`); total params **3.73B**, active **0.91B ≈ 1B dense-equivalent**.