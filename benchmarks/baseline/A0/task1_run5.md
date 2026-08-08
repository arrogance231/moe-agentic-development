# Archi — 1B Dense-Equivalent MoE Architecture

## 1. Overview

A decoder-only transformer with **dense-equivalent FLOPs ≈ a 1B dense model** (≈1.01B active params per token) but **≈2.8B total parameters**, following the Switch/ST-MoE style of replacing every FFN with a shared MoE layer. All non-FFN blocks (embeddings, attention, router) are dense and always active; only **2 of 8 experts** are active per token. Compute-per-token is pinned to the ~1B dense budget while parameter memory is expanded ~2.8×.

## 2. Parameters

Config: `d_model = 2048`, `num_layers = 18`, `num_experts = 8`, `top_k = 2`, `expert_ffn_dim = 4096`, `vocab = 50,000`.

| Component | Shape | Params (per instance) | Total (all layers) | Active per token |
|---|---|---|---|---|
| Token embedding | 50,000 × 2,048 | 102.4M | 102.4M | 102.4M |
| Attention ×18 | 4 × 2,048² | 16.78M | 302.0M | 302.0M |
| Expert FFN ×18 × 8 | 2 × 2,048 × 4,096 | 16.78M/expert | 2,415.9M | 603.9M (2 of 8) |
| Router ×18 | 2,048 × 8 + 8 | 16.4K | 0.3M | 0.3M |
| **Total** | | | **2,820.7M (2.82B)** | **1,008.6M (1.01B)** |

**Key figures:** `num_experts = 8`, `top_k = 2`, `total_params = 2.82B`, `active_params = 1.01B`, sparsity ratio = 2.80×.

**Parameter math (explicit):**
```
Attention/layer = 4·d_model²            = 4·2048²       = 16,777,216
Expert FFN/layer = E·2·d_model·d_expert = 8·2·2048·4096 = 134,217,728
Per-layer total  = 16.78M + 134.22M + 0.02M ≈ 151.0M
Model total      = 18·151.0M + 102.4M (embed) = 2,820.7M
Active experts/layer = top_k·2·d_model·d_expert = 2·16.78M = 33.55M
Active params   = 102.4M + 18·16.78M + 18·33.55M + 0.3M ≈ 1,008.6M
```

## 3. Routing Choice

**Learned softmax top-2** — a per-layer linear router (dense 2048→8 logits) followed by a softmax; the two highest-probability experts receive the token, and outputs are weighted by their softmax probabilities.

- **top-2 over top-1:** two experts per token nearly halves router error sensitivity, smooths gradients to more experts, and yields better expert utilization/representation diversity (ST-MoE: top-2 ≥ top-1 quality for similar compute). Top-1 is simpler but more load-imbalance-prone.
- **top-2 over "all experts":** activating all experts restores dense FLOPs and destroys the compute-equivalence property — the entire point of the 1B-equivalent budget.
- **Learned over random/hash routing:** learned routing adapts expert specialization to data; hash routing is cheaper and allocation-free but gives measurably worse quality on next-token prediction.
- **Soft vs hard:** the weighted (soft) combination of expert outputs is differentiable and standard; pure hard top-k argmax would need straight-through or EM tricks.

**Stability guard:** the router is dense (fully connected to the hidden state), so gradients flow to every expert through the top-2 probabilities — no disconnected experts, unlike hard top-1.

## 4. Training Implications

- **Capacity factor = 1.25.** Per expert, per layer, the token buffer is `ceil((batch_tokens / E) · 1.25)`. The 25% headroom absorbs routing skew; excess tokens are dropped (Switch-style). 1.0 is too brittle; ≥2.0 wastes compute.
- **Auxiliary load-balancing loss** (Switch Transformer): `L_aux = α·E·Σᵢ fᵢ·Pᵢ`, where `fᵢ` = fraction of tokens routed to expert `i`, `Pᵢ` = mean router probability for expert `i`, **α = 0.01**. This is the standard, cheap-to-compute, fully-differentiable balance regularizer.
- **Batching:** with E=8, each expert needs a large token batch; use a large global batch (≥0.5M tokens) and gradient accumulation so per-expert batches stay statistically meaningful.
- **Parallelism:** expert-parallel all-to-all dispatch per layer; communication cost is per-layer, so it is amortized by deep batches. FSDP/ZeRO with sharded optimizer states (Adam: 2× params = 5.6B of optimizer state → use 8-bit Adam / offload) is required for 2.82B params.
- **Optimization:** smaller LR than an equal-FLOPs dense model (higher total params ⇒ higher gradient variance), longer warmup, gradient clipping; the auxiliary loss couples all experts so LR must be conservative.
- **FLOPs:** per-token FLOPs ≈ a dense 1B model (2·active_params·tokens ≈ 2.0 GFLOPs/token), so training throughput should roughly match a 1B dense baseline *modulo* all-to-all overhead.

## 5. Risks

- **Expert collapse / load imbalance:** one or two experts absorb most tokens despite the aux loss; the rest are dead weight. Mitigated by α=0.01 tuning (raise if imbalance, lower if quality degrades).
- **Capacity overflow ⇒ token dropout:** at CF=1.25, skewed routing silently drops tokens, biasing the distribution. Requires monitoring per-expert utilization each step.
- **Router instability early in training:** router gradients are noisy before experts differentiate; mitigate with longer warmup and router entropy monitoring.
- **Communication-bound training:** all-to-all per layer can dominate wall-clock on small/wide batches; real throughput may be below the 1B dense-equivalent FLOPs projection.
- **Memory-bound inference:** all 8 experts (~2.4B params) must reside in memory even though only 2 are active; little benefit on latency-critical or memory-constrained deployment.
- **Sample-efficiency gap:** 2.8B total params on a fixed data budget overfits more easily than a 1B dense model; requires sufficient data (≈≥ 200B tokens) for the expert capacity to pay off.
- **Quality regression at this scale:** MoE gains are notoriously small at <1B active scale (overhead eats the benefit); the design banks on scale-up and must be validated against a dense 1B baseline early (e.g., at 30% of training).