# 1B Dense-Equivalent MoE — Architecture Design

## Overview

This document specifies a sparse Mixture-of-Experts (MoE) transformer whose **total parameter count ≈ 1.05B** — the "dense-equivalent" size (a standard dense model of this scale has all params active every token). We use a learned, soft **top-2** router with a load-balancing auxiliary loss. Per token only ~**228M** params are active, so per-token FLOPs are ≈ 4.6× lower than a dense 1B model, matching the inference-compute profile of a ~0.23B dense model.

**Design intent:** a 1B-class parameter footprint (competitive with dense 1B models) at ~¼ the per-token compute, suitable for single-GPU inference and mid-scale training clusters.

## Parameters

| Component | Formula | Params |
|---|---|---|
| Embeddings (tied w/ LM head, vocab 50,000) | 50,000 × 1,024 | 51,200,000 |
| Per layer — Attention (Q/K/V/O) | 4 × 1,024² | 4,194,304 |
| Per layer — Router | 1,024 × 16 | 16,384 |
| Per layer — Experts (16 × FFN) | 16 × 2 × 1,024 × 2,048 | 67,108,864 |
| Per layer — LayerNorm (4×) | 4 × 1,024 | 4,096 |
| **Per layer total** | — | **71,323,648** |
| 14 layers | 71,323,648 × 14 | 998,531,072 |
| **Total (dense-equivalent)** | 998,531,072 + 51,200,000 | **≈ 1,049,731,072 ≈ 1.05B** |
| **Active params per token** | 51.2M + 14 × (4.19M + 8.39M + 0.02M) | **≈ 228M** |

**Explicit model figures**

| Hyperparameter | Value |
|---|---|
| **Total parameters** | **≈ 1.05B** |
| **num_experts** | **16** |
| **top_k** | **2** |
| Hidden size `d_model` | 1,024 |
| Expert FFN hidden `d_ff` | 2,048 |
| Layers | 14 |
| **Routing strategy** | learned softmax **top-2** |
| **Capacity factor** | **1.25** |
| **Auxiliary loss** | Switch-style load balance, α = 0.01 |

**Capacity math** (per layer, per batch of N tokens): capacity = ⌈1.25 × top_k × N / E⌉ = ⌈1.25 × 2 × N / 16⌉. For N = 2,048 → ⌈0.15625 × 2,048⌉ = **320 tokens/expert** (25% slack over the ideal 256).

## Routing choice

**Choice: learned, soft top-2 (weighted softmax gating) + auxiliary load-balancing loss.**

| Option | Verdict |
|---|---|
| Learned top-2 (chosen) | Better representational capacity than top-1 (Mixtral-style); standard, differentiable, easy to train |
| Top-1 / Switch | Simpler + 2× less FFN compute, but 2× fewer experts combine per token → lower quality per unit param |
| Fixed hash/random (GShard) | Zero router params, no aux loss needed, but no specialization and measurably worse perplexity |
| Hard (argmax) selection | Undifferentiable w.r.t. router → must resort to score-approximation; not worth it |
| Soft (probabilistic) selection | Larger variance / memory for marginal quality gains; **soft = softmax-weighted mixture** chosen instead |

Justification: top-2 gives 2× the experts-per-token of Switch while only ~2× the FFN FLOPs (a fair trade at this scale), and the **soft weighted combination** (expert outputs scaled by their softmax probabilities) keeps the gate differentiable end-to-end. The learned router is a single linear layer (`d_model → E`) — negligible 16K params/layer — and is regularized by the auxiliary loss to prevent collapse.

## Training implications

- **Throughput:** ~4.6× fewer active params → ~4.6× more tokens/sec at fixed FLOP budget; must scale batch tokens to keep effective batch sizes (esp. on the router's gradients) comparable to a dense 1B run.
- **Load balancing:** capacity factor 1.25 lets routing drift within 25% of uniform before token dropping; the α=0.01 aux loss (Σ fᵢ·Pᵢ, scaled by E) gently pulls the router toward uniform assignment. Tune α: too high → uniform-but-useless experts; too low → imbalance → dropped tokens.
- **Token dropping:** when an expert exceeds capacity, overflow tokens are dropped (token budget is what pays for compute savings) — set capacity slack to keep drop rate < ~1–2%.
- **Distributed training:** experts are sharded across devices (expert parallelism); tokens must be dispatched via all-to-all, so communication latency, not raw FLOPs, becomes the bottleneck at small batch sizes.
- **Optimization:** slightly longer warmup than dense (router gradients are noisier early), and a lower peak LR is typical; the aux loss adds a small effective regularization term.
- **Quality guardrail:** aux loss + capacity slack together bound the quality loss versus a dense 1B baseline, which is the reference target.

## Risks

- **Router collapse / degenerate specialization:** all tokens funnel to a few experts → most experts idle. Mitigated by the aux loss and monitored via per-expert token histograms.
- **Load imbalance & dropped tokens:** with capacity 1.25, tail-heavy distributions silently drop information; monitor drop rate and raise capacity factor if >2%.
- **Aux loss interference:** an over-strong aux loss degrades task quality (uniform ≠ good routing); keep α small and validate on downstream evals, not just perplexity.
- **Communication overhead:** at small per-expert batch sizes, all-to-all dispatch can eat the FLOPs savings — particularly on low-bandwidth multi-node setups.
- **Under-specialization at 1B scale:** 16 experts over a 1B model can produce redundant, poorly specialized experts, wasting parameters; consider fewer experts (e.g., 8) if evals plateau.
- **Overfitting / instability:** MoE at small scale + small datasets can overfit the router; use dropout / weight decay on the gate and monitor training-vs-eval divergence.
- **Per-token FLOPs assumption:** the 4.6× compute saving only holds if capacity slack stays near 1.0 — raised capacity or heavy top-2 recomputation erodes the win.