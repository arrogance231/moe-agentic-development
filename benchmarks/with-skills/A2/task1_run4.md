# MoE Architecture — 1B Dense-Equivalent

## Overview

This design replaces the FFN of a ~1.14B dense transformer with a Mixture-of-Experts block: **32 experts, top-2 active, half-width expert FFNs**. Because `top_k × expert_FFN = dense_FFN` exactly, the activated parameters (1.14B) and FLOPs/token (7.1 GFLOPs) match the dense baseline while total parameters grow ~11.6× to **13.22B**. Routing is Top-2 learned with capacity factor **1.25** and auxiliary loss **0.01**. Assumed spec: `num_layers=16, d_model=2048, dense ffn_mult=4 (ffn_dim=8192), vocab=32000, seq_len=2048`.

## Parameters (math shown)

```
ffn_dim (dense)   = 4 × 2048                    = 8192
expert ffn_dim    = 4096                        (half-width expert)
attention/layer   = 4 × 2048²                   = 16,777,216
layernorm/layer   = 2 × 2048                    = 4,096
dense FFN/layer   = 3 × 2048 × 8192             = 50,331,648
expert FFN        = 3 × 2048 × 4096             = 25,165,824
embedding         = 32,000 × 2,048              = 65,536,000
```

| Component (all layers) | Dense | MoE | Activated (top-2) |
| --- | --- | --- | --- |
| Attention (16 × 16,777,216) | 268,435,456 | 268,435,456 | 268,435,456 |
| Expert FFNs | 805,306,368 | 12,884,901,888 | 805,306,368 |
| | (16 × 50,331,648) | (16 × 32 × 25,165,824) | (16 × 2 × 25,165,824) |
| Layernorms (16 × 4,096) | 65,536 | 65,536 | 65,536 |
| Embedding | 65,536,000 | 65,536,000 | 65,536,000 |
| **Total** | **1,139,343,360** | **13,218,938,880** | **1,139,343,360** |

```
dense_params  = 16·(16,777,216 + 50,331,648 + 4,096) + 65,536,000 = 1,139,343,360  (1.14B)
moe_params    = 16·(16,777,216 + 4,096 + 32·25,165,824) + 65,536,000 = 13,218,938,880  (13.22B)
activated     = 16·(16,777,216 + 4,096 + 2·25,165,824) + 65,536,000 = 1,139,343,360  (1.14B)
param_ratio   = 13,218,938,880 / 1,139,343,360  = 11.60
flops/token   = 6·1,139,343,360 + 4·16·2048·2048 = 7,104,495,616  (~7.10 GFLOPs, = dense)
```

**num_experts = 32, top_k = 2.** Total = **13.22B params**; activated = **1.14B** (= dense baseline; ratio 1.0).

## Routing choice

- **Strategy: Top-2 learned router** (default for quality-focused training). Justification: two half-width experts per token restore the dense FFN capacity *and* blend expert knowledge; Top-1 would halve activated expert capacity and drop quality at this scale.
- **Capacity factor: 1.25** — absorbs token-routing imbalance (up to 25% headroom) without wasting compute; avoids token drops during early imbalance.
- **Auxiliary loss: 0.01** — the upper end of the 0.001–0.01 band is warranted because 32 experts at 1B scale are prone to routing collapse; keeps the effective expert count high. Router weights (~65K/layer) are negligible and omitted.

## Training implications

- **Compute:** 7.10 GFLOPs/token — identical to the dense 1.14B baseline, so training cost per token is unchanged; only memory grows.
- **Memory:** 13.22B total params (~26.4 GB BF16, plus ~158 GB AdamW-mixed optimizer states) vs 1.14B activated. The expert set (12.9B params) must be sharded with expert parallelism; dense parts can use ZeRO.
- **Parallelization:** `DP × TP × PP × EP = #GPUs`; **EP must divide 32** (e.g. EP=8 on 8 GPUs → 4 experts/rank). Each MoE layer incurs one all-to-all per forward/backward (16 layers), so dispatch volume = `top_k × tokens × bytes × EP-group-size`.
- **Batch:** target ≥8–64 tokens/expert/GPU per micro-batch (`micro_batch × 2048 × 2 / (32/EP)`); use BF16; gradient checkpointing if seq_len > 2048.

## Risks

- **Over-parameterization without compute gain** (13.2B total for 1B compute): inflates checkpoint and optimizer memory. *Mitigation:* ZeRO + EP sharding, BF16, eval-time expert pruning.
- **All-to-all bottleneck:** 32 experts × 16 MoE layers. *Mitigation:* pick EP = GPU count, overlap comm/compute, drop to 16 experts on small clusters.
- **Routing collapse / imbalance:** likely without strong load balancing. *Mitigation:* aux loss 0.01, capacity 1.25, jittered logits at train time.
- **Half-width experts under-capacity:** each expert alone is weaker than a dense FFN. *Mitigation:* top-2 blending; verify effective-expert count ≥ 16 during training.

All figures above are derived directly from the skill's formulas (`ffn_dim`, per-layer attention, GLU expert FFN, embedding, dense/MoE/activated totals, `flops_per_token`, `param_ratio`).