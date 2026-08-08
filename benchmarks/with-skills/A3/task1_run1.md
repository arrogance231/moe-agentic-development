Let me check for the calculator tool referenced by the skill and look at the workspace.
# MoE Architecture: 1B Dense-Equivalent Model

## Overview

This design replaces the FFN blocks of a 653M-parameter dense model with Mixture-of-Experts layers: **16 experts, top-2 routing**. Because top-2 activates two experts per token, the model touches **≈1.11B parameters per token**, giving training compute equivalent to a **~1B dense model** (the "dense-equivalent" figure) while storing **7.45B total parameters**. The activated/expert capacity (2 of 16 experts) is the standard GShard-style training sweet spot. All numbers below recompute exactly with the `moe-architecture` formulas as implemented in `tools/moe_calculator.py`.

**Design spec**

| Field | Value |
| --- | --- |
| num_layers | 16 |
| d_model | 1536 |
| ffn_mult / ffn_dim | 4 / 6144 |
| vocab | 32,000 |
| seq_len | 2,048 |
| **num_experts** | **16** |
| **top_k (active experts)** | **2** |
| capacity_factor | 1.25 |
| aux_loss (router) | 0.01 |

## Parameters

**Math** (per the calculator formulas, `ffn_dim = 4 · 1536 = 6144`):

```
attention/layer = 4·d²          = 4·1536²         = 9,437,184
dense FFN/layer = 3·d·ffn_dim   = 3·1536·6144     = 28,311,552
layernorm/layer = 2·d           = 2·1536          = 3,072
embedding       = vocab·d       = 32,000·1536     = 49,152,000

dense   = 16·(9,437,184 + 28,311,552 + 3,072) + 49,152,000
        = 604,028,928 + 49,152,000 = 653,180,928
moe     = 16·(9,437,184 + 3,072 + 16·28,311,552) + 49,152,000
        = 7,398,801,408 + 49,152,000 = 7,447,953,408
activated = 16·(9,437,184 + 3,072 + 2·28,311,552) + 49,152,000
        = 1,057,013,760 + 49,152,000 = 1,106,165,760
```

| Component | Dense | MoE (total) | Activated (per token) |
| --- | --- | --- | --- |
| Attention (16 layers, 16·9,437,184) | 150,994,944 | 150,994,944 | 150,994,944 |
| Layernorms (16 layers, 16·3,072) | 49,152 | 49,152 | 49,152 |
| Expert FFNs (16 layers; 1 / 16 / 2 experts·28,311,552) | 452,984,832 | 7,247,757,312 | 905,969,664 |
| Embedding (vocab·d) | 49,152,000 | 49,152,000 | 49,152,000 |
| **Total** | **653,180,928** | **7,447,953,408** | **1,106,165,760** |

- **Total parameters: 7,447,953,408 (7.45B)**
- **Activated parameters: 1,106,165,760 (1.11B) ≈ 1B dense-equivalent**
- **num_experts = 16, top_k = 2**
- param_ratio (MoE/dense) = 11.4×; total/activated = 6.7×

## Routing choice

**Top-2** (learned router with softmax over expert logits).

Justification: this is a **training-quality-focused** design, not latency-bound inference, so Top-2's only cost — one extra expert dispatch per token (≈2× all-to-all vs Top-1) — is paid on GPU clusters where it is cheap. Top-2 is the standard training default and gives a large quality gain over Top-1 by blending two experts' knowledge while staying far more stable than learned/soft routing at this expert count. (Top-1 is rejected: it halves active expert capacity; learned/soft is rejected: 16 experts is too many for it to train stably.)

- **capacity_factor = 1.25** — absorbs early-training token-routing imbalance without dropping tokens (1.0 risks overflow/drop during router warmup).
- **aux_loss = 0.01** — upper end of the 0.001–0.01 band; guards the main small-scale failure mode (router collapse), cheap to carry since 1.25 capacity makes mild over-regularization harmless.

## Training implications

**Compute.** Training FLOPs ≈ 6·activated + attention term:

```
flops/token = 6·1,106,165,760 + 4·16·1536·2048
            = 6,636,994,560 + 201,326,592 = 6,838,321,152 ≈ 6.84 GFLOPs/token
```

vs 4.12 GFLOPs/token for the 653M dense baseline (+66%), but ≈ a true 1B dense model's compute — FLOPs scale with the 1.11B activated count, not the 7.45B total.

**Memory** (8× A100-40GB, BF16, EP=8, DP=1; or EP=4/DP=2 — EP must divide the 16 experts):

| Component | Total | Per GPU (8 ranks) |
| --- | --- | --- |
| Parameters (BF16) | 14.9 GB | ~1.9 GB (experts sharded by EP, dense via ZeRO) |
| Gradients | 14.9 GB | ~1.9 GB |
| Optimizer (AdamW, 12 B/param) | 89.4 GB | ~11.2 GB |
| Activations (micro-batch 2, seq 2048) | — | ~2 GB |
| Overhead | — | ~10% + 1.5 GB |
| **Total** | | **~18 GB/GPU (< 40 GB, >50% headroom)** |

**Parallelization.** One all-to-all per MoE layer (16 layers) at EP=8 → 2 experts/rank. Tokens per expert per micro-batch ≈ `2·2048·2/(16/8) = 4096`, far above the 8–64 utilization floor, so expert utilization is healthy even at micro-batch 2; capacity 1.25 keeps dispatch volume ~25% above ideal.

**Checkpointing.** BF16 + `safetensors`; keep capacity_factor and aux_loss consistent with this document (1.25 / 0.01).

## Risks

- **Over-parameterization without compute gain** — 7.45B total vs 1.11B activated inflates checkpoint size and optimizer memory. **Mitigation:** EP=8 sharding + ZeRO keeps per-GPU memory at ~18GB; expert count capped at 16 so the ratio stays 6.7×, not the 20–30× that 64+ experts would produce at this scale.
- **Routing collapse** — router sends all tokens to a few experts, nullifying the 16-expert investment. **Mitigation:** aux_loss 0.01 (top of band) + capacity 1.25 keeps under-loaded experts fed; monitor per-expert counts and effective-expert share during training.
- **All-to-all communication** — 16 expert layers × top-2 dispatch. **Mitigation:** capacity 1.25 (not 2.0) bounds dispatch volume; overlap comm/compute; verify EP=8 vs EP=4 step times if comm-bound.
- **Top-2 vs Top-1 quality/cost** — 2× dispatch vs Top-1 is accepted as the price of training quality; revisit Top-1 only if a deployable inference latency target appears.
- **Embedding term** — 49M is ~7% of activated params; included correctly in all three totals, so nothing is mis-estimated.