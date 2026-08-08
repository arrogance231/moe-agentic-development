Let me check for the calculator tool referenced by the skill.
I have enough to design this. Note: no calculator executable is available in this environment, so I derive every number from the skill's formulas and show the arithmetic explicitly for cross-checking. The design mirrors a published 1B-scale MoE (`large-moe-1b`: 16 layers, d=1024, 8 experts, top-2, ≈1.0B total / 0.4B active).

---

# Architecture: 1B dense-equivalent MoE

## Overview

This design replaces a **1.01B dense transformer** with an MoE variant of **equal total parameter count** (the "dense-equivalent" reading used by Switch Transformer: the MoE stores as much as the dense model it replaces, but activates only a fraction per token). The MoE keeps the same attention stack, depth, and width as the dense baseline and only converts the FFN into a **8-expert, Top-2** mixture. Total parameters are **1.014B**, of which only **0.335B (33%)** are activated per token, cutting training FLOPs/token to **~0.35× the dense baseline** at equal memory footprint. A small vocab (32k) keeps the embedding term in check, as it is a significant share of parameters at this scale.

| Headline | Value |
| --- | --- |
| **Total parameters** | **1,014,272,000 (≈ 1.014B)** |
| Dense baseline (matched total) | 1,014,272,000 (1.014B) |
| Activated parameters (per token) | 334,794,752 (0.335B, 33.0%) |
| **num_experts** | **8** |
| **top_k** | **2** |
| d_model / num_layers / vocab / seq_len | 1024 / 18 / 32,000 / 2048 |
| Expert FFN width (GLU) | 2,048 |
| param_ratio (MoE/dense) | 1.00 |

## Parameters

Formulas (from `moe-architecture` skill; arithmetic shown, all recomputed):

```text
attention (per layer)   = 4·d²                    = 4·1024²       = 4,194,304
expert FFN (GLU, e=1)   = 3·d·ffn_dim_e           = 3·1024·2048   = 6,291,456
8-expert FFN (per layer)= 8 × 6,291,456           = 50,331,648
dense FFN (per layer)   = 3·d·ffn_mult·d, mult=16 = 3·1024·16,384 = 50,331,648   (equal to 8-expert total — this is the dense-equivalence point)
LayerNorm (per layer)   = 2·d                     = 2,048
embedding               = vocab·d                 = 32,000·1024   = 32,768,000
router                  = n_experts·d             = 8·1024 = 8,192/layer → 147,456 total (0.01%, omitted)
```

| Component (18 layers) | Dense | MoE | Activated (top-2) |
| --- | ---: | ---: | ---: |
| Attention | 18 × 4,194,304 = 75,497,472 | 75,497,472 | 75,497,472 |
| FFN / Expert FFNs | 18 × 50,331,648 = 905,969,664 | 18 × 50,331,648 = 905,969,664 | 18 × 12,582,912 = 226,492,416 |
| LayerNorms | 18 × 2,048 = 36,864 | 36,864 | 36,864 |
| Embedding | 32,000 × 1,024 = 32,768,000 | 32,768,000 | 32,768,000 |
| **Total** | **1,014,272,000** | **1,014,272,000** | **334,794,752** |

Check: MoE total = 75,497,472 + 905,969,664 + 36,864 + 32,768,000 = **1,014,272,000**; activated = 334,794,752 (**33.0%** of total); param_ratio = 1.00.

## Routing choice

- **Strategy: Top-2.** The default for training; it retains knowledge blending across two experts and consistently beats Top-1 at equal capacity, at a modest all-to-all cost (2 dispatches per layer). Top-1 is only preferred for latency-bound inference, which is not the goal here.
- **top_k = 2** with 8 experts gives sparsity factor 4 (activated expert width 2×2,048 = 4,096 vs 16,384 total), a healthy balance between capacity and compute saving.
- **Capacity factor: 1.25.** Absorbs token-routing imbalance without dropping tokens (1.0 would drop under skew); 1.25 wastes only modest compute on padding.
- **Auxiliary load-balancing loss: 0.01.** Upper end of the 0.001–0.01 band; at 8 experts with a 32k-vocab router the risk of collapse is real, and 0.01 keeps utilization high without visibly distorting the routing objective.

## Training implications

- **Compute:** FLOPs/token = 6·activated + 4·layers·d·seq = 6×334,794,752 + 4×18×1024×2048 = 2,008,768,512 + 150,994,944 = **2.16 GFLOP/token** vs dense **6.24 GFLOP/token** → **0.35×** the compute of the matched dense model. The same GPU budget can process ~2.9× more tokens.
- **Memory:** total 1.014B params dominate storage, not the 0.335B activated. On 8× H100-80GB with DP=2 × EP=4 (experts sharded 2 per rank), per-GPU estimate: params+grads ≈ 0.7 GB bf16 each, AdamW optimizer ≈ 4 GB, activations (micro-batch 8, seq 2048, with recompute) ≈ 3 GB → ≈ 11 GB/GPU, > 80% headroom. Tokens per expert per micro-batch = 8×2048×2/8 = 4096, well above the 8–64 utilization floor.
- **Parallelization:** EP=4 divides 8 experts; one all-to-all per MoE layer (18 per step). Top-2 keeps dispatch volume modest; capacity 1.25 bounds it.

## Risks

1. **Router collapse / expert imbalance.** A weak aux loss at small expert counts starves experts. Mitigation: aux loss 0.01, capacity 1.25, add router-logit jitter at train time; monitor effective experts via router distribution.
2. **Token dropping under imbalance.** At capacity 1.25, overflow only occurs under heavy skew; if overflow fraction is non-trivial, raise capacity or strengthen the aux loss rather than raising capacity blindly.
3. **Active-capacity quality risk (most relevant here).** Activated params are 0.33B vs the 1B dense — per-token compute is 1/3 of dense. If quality under-delivers, raise top_k to 3–4 (sparsity 8/3 ≈ 2.7) or raise expert FFN width before adding experts.
4. **All-to-all overhead at scale.** Every extra expert/top-k raises dispatch volume; 8×Top-2 is safe on NVLink/IB, but do not push expert count without raising EP.
5. **Embedding term at small scale.** vocab·d = 32.8M is 3.2% of params; growing vocab to 150k would push the embedding to ~155M (15%) and bias the parameter math — keep the embedding included in all budgets.