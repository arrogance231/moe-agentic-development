# MoE-1B: A 1B Dense-Equivalent Sparse MoE Architecture

## Overview

MoE-1B is a transformer-based, Mixtral-style sparse Mixture-of-Experts language model. It carries **~1.04B total parameters** (the "dense-equivalent" figure) but only activates **~338M parameters per token** by routing each token through a learned top-2 of 16 experts per layer. The model keeps a standard dense attention stack (all tokens pass through every layer's self-attention) and replaces each MLP block with an expert FFN pool + router. This yields the representational capacity of a ~1B model at roughly the per-token compute of a ~0.34B dense model.

Design targets: 1B total params, stable training (z-loss + load-balancing auxiliary loss), expert-parallel friendly (16 experts is a practical count for 8–32-way parallelism), and quality headroom via a 1.25 capacity factor with token dropping as a safeguard.

## Parameters

| Component | Formula | Value | Digits |
|---|---|---|---|
| Embedding (tied in/out) | `V × d = 50,304 × 2048` | 103,022,592 | 103.0M |
| Attention per layer | `4 × d² = 4 × 2048²` | 16,777,216 | 16.78M |
| Attention total (8 layers) | `16.78M × 8` | 134,217,728 | 134.2M |
| Expert FFN per expert | `3 × d × e = 3 × 2048 × 1024` (gate/up/down) | 6,291,456 | 6.29M |
| Experts per layer | `E = 16` | — | — |
| Expert FFN per layer | `6.29M × 16` | 100,663,296 | 100.7M |
| Expert FFN total (8 layers) | `100.7M × 8` | 805,306,368 | 805.3M |
| Router per layer | `d × E = 2048 × 16` | 32,768 | 0.03M |
| Router total | `32,768 × 8` | 262,144 | 0.3M |
| LayerNorm (8 layers × 3 norms) | `8 × 3 × 2 × 2048` | 98,304 | 0.1M |
| **Total parameters** | sum of above | **1,042,907,136** | **≈ 1.043B** |
| Active params/token | emb + 8×(attn + `2×6.29M` + router) + norms | ~338M | ≈ 0.34B |

Config summary: `vocab=50,304` · `hidden=2048` · `layers=8` · `heads=16` (head_dim 128) · **`num_experts=16`** · **`top_k=2`** · expert FFN dim `1024` · tied embeddings.

Explicit total math: `103.02M + 134.22M + 805.31M + 0.26M + 0.10M ≈ 1.043B`. Active math per token: `103.0M + 8 × (16.78M + 12.58M + 0.03M) + 0.1M ≈ 338M`.

## Routing Choice

- **Strategy:** learned token-level softmax routing over the top-2 experts (Mixtral-style gating). Each token computes `g = softmax(W_r · x)`, takes the two largest logits, and weights expert outputs by the softmax probabilities.
- **Chosen: top-2 / learned / soft.** Rationale:
  - *top-2 over top-1:* two experts per token smooth the router gradient, spread load, and avoid the router-collapse/dead-expert failure mode that plagues aggressive top-1 sparsity. top-1 is cheaper (fewer active FLOPs) but is the regime most prone to load collapse and per-token quality loss in sub-10B-scale models.
  - *top-2 over dense-soft (all-expert mixing):* "soft routing" without a top-k makes every token touch all 16 experts, destroying the MoE compute savings and turning the layer into an expensive dense FFN. Keeping hard sparsity (`k=2`) is what preserves the ~338M active vs ~1.04B total gap.
  - *learned over static/fixed routing:* learned token-level routing lets the model specialize experts by domain/syntactic role with no hand-crafted assignment, and the router adds only `d × E = 32,768` params per layer (~0.3M total), negligible against the 805M expert block.

## Training Implications

- **Compute profile:** ~338M active params/token ≈ per-token FLOPs of a ~0.34B dense model, so training compute is *lower* than a real 1B dense baseline, despite more total memory. This is the core MoE trade-off: capacity for free, memory and bandwidth are not.
- **Memory & parallelism:** all 16 experts/layer must live in memory (805M params). Requires expert parallelism: experts sharded across devices, tokens dispatched via all-to-all (or sequence parallel) every MoE layer. 16 experts maps cleanly to 8–16-way EP.
- **Communication:** each MoE layer performs a full all-to-all (gather-dispatch). Communication cost grows with `E` and batch/sequence length; on low-bandwidth or high-latency clusters the all-to-all can dominate wall-clock and negate the FLOP advantage.
- **Batch size / load balance:** router statistics need enough tokens per batch to converge — use large effective batches (gradient accumulation or larger global batch) so the auxiliary loss has signal.
- **Hyperparameters chosen (per requirements):**
  - Expert count: **16** per layer
  - Active experts: **top-2**
  - Routing: **learned softmax, top-2 gating**
  - Capacity factor: **1.25** (per-expert buffer = `CF × tokens_per_seq / E`; overflowed tokens are dropped, and their residual bypasses the FFN via skip connection)
  - Auxiliary loss: **load-balancing aux loss, coefficient 0.01** (Switch/ST-MoE formulation) **+ router z-loss, coefficient 0.001** (caps router logit growth for stability)

## Risks

- **Load imbalance / router collapse:** top-2 softmax can still drift toward a few popular experts, producing dead experts and wasted capacity. Mitigated by the 0.01 aux loss; tuning the coefficient is the single most sensitive knob.
- **Aux-loss quality tax:** a too-high aux weight forces uniform routing even when concentration would be loss-optimal, hurting final loss; too-low risks collapse. Needs monitoring of per-expert token counts and router entropy.
- **Token dropping:** with CF=1.25, extreme batches can drop tokens on busy experts, silently degrading quality on that data (typically long-tail/rare tokens). CF must be validated empirically.
- **Training instability:** sparse MoE models show loss spikes in early training, especially at small scale and with k>1; the z-loss mitigates but spikes may still require gradient clipping and LR warmup.
- **Communication-bound training:** on multi-node clusters with slow interconnects, all-to-all cost per MoE layer can dominate, making the effective MFU much lower than a dense model of equal active size.
- **Data-hunger:** 1.04B total params with only ~338M active means many more parameters must be trained per token; expect a *higher* token budget (and possibly a degradation floor) relative to a pure dense 1B on small datasets — MoE pays off when data is plentiful.
- **Serving complexity:** 16× parameters per MoE layer must be resident on GPUs (or aggressively offloaded), increasing host memory; per-token latency depends on expert locality and routing fan-out.