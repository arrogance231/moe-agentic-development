# MoE-1B: A 1B Dense-Equivalent Sparse Mixture-of-Experts Language Model

## Overview

This document specifies a decoder-only transformer in which every Feed-Forward Network (FFN) layer is replaced by a Sparse Mixture-of-Experts (SMoE) layer. The model is engineered so that **active parameters per token ≈ 1.0B** — i.e., its per-token compute cost is equivalent to that of a ~1B dense model — while storing ~2.2B parameters total. This is the classic MoE trade: pay the FLOPs of a 1B model, retain the capacity of a 2.2B model.

Design is Mixtral-style: coarse-grained experts (8 per layer), learned **top-2** softmax routing, with Switch-style load-balancing auxiliary loss and a capacity factor to bound per-expert work.

## Parameters

| Component | Per-layer params | × Layers | Total params | Active per token |
|---|---|---|---|---|
| Token embedding (50,000 × 2048) | — | — | 102.4 M | 102.4 M |
| Attention (Q, K, V, O, bias-free) | 4 × 2048² = 16.78 M | 24 | 402.7 M | 402.7 M |
| Router (2048 × 8, bias-free) | 16,384 | 24 | 0.39 M | 0.39 M |
| Experts — 8 × GELU FFN (2 × 2048 × 2048) | 67.11 M | 24 | 1610.6 M | 402.7 M (= 2/8) |
| LM head (2048 × 50,000, untied) | — | — | 102.4 M | 102.4 M |
| **Total** | | | **≈ 2.22 B** | **≈ 1.01 B** |

Key figures: **total parameters ≈ 2.22B**, **num_experts = 8**, **top_k = 2**, sparsity ratio (total/active) ≈ **2.2×**.

**Explicit math:**

- Expert params, one layer: `E × 2 × d_model × d_ff = 8 × 2 × 2048 × 2048 = 67,108,864` (each expert = one up-projection 2048→2048 and one down-projection 2048→2048)
- Active expert params per token, one layer: `(k/E) × 67.11 M = (2/8) × 67.11 M = 16.78 M` — exactly equal to the attention block, by construction
- Active params total: `24 × (16.78 M attn + 16.78 M experts + 16 k router) + 102.4 M emb + 102.4 M head = 1010.6 M ≈ 1.0 B` ✓
- Total params: `24 × (16.78 M + 67.11 M) + 204.8 M = 2218.5 M ≈ 2.22 B`

## Routing Choice

- **Strategy:** learned linear router (one weight matrix per layer) producing logits → `softmax` over all 8 experts, then **top-2** selection; the two selected weights are renormalized and the layer output is the weighted sum of the two expert outputs: `y = Σ_{i∈top2} w_i · FFN_i(x)`.
- **Top-2 over top-1:** top-1 (Switch Transformer) is cheapest but provokes load imbalance and router collapse; top-2 gives each token a second, weakly-specialized expert, smoother gradients, redundancy, and materially better quality at only 2× expert FLOPs — still 4× cheaper than activating all 8. This is the Mixtral-validated recipe.
- **Learned over fixed/hash routing:** a learned gate adapts specialization to the data distribution; hash/random routing cannot learn which inputs share structure. **Soft (weighted) over hard (argmax):** softmax weights keep the router differentiable and let output blend expertise; hard routing is non-differentiable and noisier.
- **Why 8 experts:** few large experts (>1B hidden each in Mixtral; here 2048-wide) beat many small ones for quality at this scale; 8 keeps the router + load-balancing tractable on small batches.

## Training Implications

- **Compute:** ~1.0B active FLOPs/token ≈ a dense 1B model; only the 2 selected experts per token receive gradients (plus the router), so sparse updates and lower training cost per token than 2.2B total suggests.
- **Load balancing & capacity:** each expert expects `T·k/E = 0.25·T` tokens per layer. **Capacity factor = 1.25** caps each expert at `⌈1.25 · T·k/E⌉` tokens; overflow tokens are dropped. Balance is enforced by an auxiliary loss: `L_aux = α·N·Σᵢ fᵢ·Pᵢ` with **α = 0.01**, where `fᵢ` is the fraction of tokens routed to expert `i` and `Pᵢ` its mean router probability (Switch). Optionally add ST-MoE router **z-loss (β = 0.001)** for training stability.
- **Memory:** all 8 experts must be resident — 2.2B params → optimizer states (AdamW, mixed precision) ≈ 8–10 GB; KV cache is dense-only (small).
- **Parallelism:** expert-parallel all-to-all dispatch per layer dominates communication; at small batch sizes the dispatch overhead can eclipse the FFN savings, so larger global batches amortize it.
- **Distribution:** per-expert mini-batch is 1/4 of global tokens; with small batches experts see noisy, high-variance gradients.

## Risks

- **Router collapse / load imbalance:** all tokens herd to a few experts; others starve and the model degrades toward dense. Mitigated by capacity factor + aux loss (α too high, however, forces uniform routing that hurts quality).
- **Communication bottleneck:** all-to-all per layer grows with tokens × d_model; on low-bandwidth interconnects MoE can be *slower* than dense despite fewer FLOPs.
- **Token dropping:** overflow beyond capacity silently drops tokens, losing information at long sequence lengths or pathological routing.
- **Training instability:** sparse routing + small per-expert batches amplify variance; needs lower LR/warmup, z-loss, and expert dropout (~0.1) to avoid divergence.
- **Memory vs. compute gap:** 2.2B total requires ~2× the memory bandwidth and VRAM of a dense 1B, complicating single-GPU deployment and KV-cache-heavy long-context serving.
- **Loss-conflict:** aux-loss pressure toward balance can conflict with the routing quality, requiring careful tuning of α and capacity as a paired knob.