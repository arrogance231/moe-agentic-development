# 1B Dense-Equivalent MoE — Architecture Document

**Design inputs:** `num_layers = 16`, `d_model = 2048`, `ffn_mult = 4` (dense), `vocab = 32,000`, `seq_len = 2048`, GLU FFN throughout. Target: replace a ~1B dense model with an MoE at the same per-token compute.

## Overview

This design replaces a **1.14B-parameter dense baseline** with a **Top-2 Mixture-of-Experts model: 32 experts, top-k=2**, where each expert FFN is half the dense FFN width (ffn_dim 4096 vs 8192). Because top-2 × half-width exactly reproduces the dense FFN compute, **activated parameters equal dense parameters to the parameter**: 1.14B activated vs 1.14B dense — a true compute-equivalent swap. Total MoE parameters grow to **13.22B** (11.6× dense) for the same FLOPs/token (7.10 GFLOP), giving a large capacity gain at unchanged training compute. Routing is Top-2 with capacity factor 1.25 and a 0.01 load-balancing aux loss.

## Parameters

| Component | Dense | MoE (all experts) | Activated (per token) |
| --- | --- | ---: | ---: |
| Attention (16 layers) | 268,435,456 | 268,435,456 | 268,435,456 |
| FFN (dense single / MoE 32 experts) | 805,306,368 | 13,153,402,880 | 805,306,368 |
| LayerNorm (16 × 2·d) | 65,536 | 65,536 | 65,536 |
| Embedding (vocab × d) | 65,536,000 | 65,536,000 | 65,536,000 |
| **Total** | **1,139,343,360** | **13,218,938,880** | **1,139,343,360** |

**Headline:** `num_experts = 32`, `top_k = 2`, total params **13.22B**, activated params **1.14B**, param ratio **11.6×**, activated ratio **1.00×**. Router weights (~`num_experts·d_model` per layer ≈ 1.0M) are negligible and omitted.

### Parameter math

Notation: `L=16`, `d=2048`, `ffn_dense = 4d = 8192`, `ffn_expert = 2d = 4096`, `E=32`, `k=2`, `vocab=32000`.

- `4·d² = 16,777,216` (attention/layer); `2·d = 4,096` (layernorm/layer)
- Dense FFN/layer: `3·d·ffn_dense = 3·2048·8192 = 50,331,648`
- Expert FFN/expert/layer: `3·d·ffn_expert = 3·2048·4096 = 25,165,824`
- Embedding: `32,000·2048 = 65,536,000`

```
dense   = L·(4d² + 3d·ffn_dense + 2d) + vocab·d
        = 16·(16,777,216 + 50,331,648 + 4,096) + 65,536,000
        = 16·67,112,960 + 65,536,000 = 1,139,343,360          (1.14B)

moe     = L·(4d² + 2d + E·3d·ffn_expert) + vocab·d
        = 16·(16,777,216 + 4,096 + 32·25,165,824) + 65,536,000
        = 16·822,087,680 + 65,536,000 = 13,218,938,880        (13.22B)

activated = L·(4d² + 2d + k·3d·ffn_expert) + vocab·d
        = 16·(16,777,216 + 4,096 + 2·25,165,824) + 65,536,000
        = 16·67,112,960 + 65,536,000 = 1,139,343,360          (1.14B = dense)

ratio   = 13,218,938,880 / 1,139,343,360 ≈ 11.6×
```

The top-2/half-width choice makes the activated per-layer FFN (`2·25,165,824 = 50,331,648`) exactly equal the dense FFN, which is why activated = dense.

**FLOPs/token** (training, forward+backward):

```
flops = 6·activated + 4·L·d·seq_len = 6·1,139,343,360 + 4·16·2048·2048
      = 7,104,495,616 ≈ 7.10 GFLOP/token   (identical to dense baseline)
```

## Routing choice

- **Strategy: Top-2** (learned linear router, hard top-2 dispatch). Default for training-quality-focused work; blends two experts per token and roughly doubles effective capacity over Top-1 at modest all-to-all cost. Top-1 would drop activated compute below the dense baseline, breaking the "dense-equivalent" property.
- **top_k = 2**, **num_experts = 32** — typical range (16–64); 32 keeps total params (13.2B) practical while giving 2/32 = 6.25% expert activation per token.
- **Capacity factor = 1.25** — absorbs token-routing imbalance without dropping tokens; 1.0 would be cheaper but drops tokens under imbalance.
- **Auxiliary loss = 0.01** (top of the 0.001–0.01 band) — at 32 experts small-scale training risks router collapse; 0.01 keeps the router load-balanced without heavily distorting the routing objective. Optional router-logit jitter at train time as a further guard.

## Training implications

- **Compute:** 7.10 GFLOP/token, identical to the dense 1.14B baseline — the MoE trains at the same per-token cost as the model it replaces, but with 11.6× the parameter capacity. Throughput-per-token parity, memory and communication are the real costs.
- **Memory (vs hardware, 8× H100 80GB):** total 13.22B params (26.4 GB bf16) + gradients + AdamW mixed optimizer states (12 B/param → 158.6 GB) dwarf the 1.14B activated footprint. Requires ZeRO-3 + expert parallelism; rough sharded budget with gradient checkpointing ≈ 32–37 GB/GPU — fits 80 GB with headroom, but the dense 1.14B would need only a few GB/GPU. Checkpoints (~26 GB bf16) are also 11.6× the dense baseline.
- **Parallelization:** `DP·TP·PP·EP = 8` with `EP | 32`. EP=8 (4 experts/rank, DP=1) or EP=4×DP=2 both valid. EP triggers **one all-to-all per MoE layer → 16 all-to-alls per step**; dispatch volume ≈ `top_k · tokens/step · dtype_bytes · EP-involved` per layer. Plan for communication/computation overlap and capacity-factor-sized per-expert activation buffers (cf 1.25 inflates buffers 25%).
- **Batch geometry:** target 8–64 tokens/expert/GPU per micro-batch; with EP=8, `micro_batch·2048·2/(32/8) = micro_batch·1024` tokens/expert/rank, so a micro-batch ≥ 4 is comfortable; gradient accumulation then sets the global batch.

## Risks

1. **Over-parameterization without compute gain (highest).** 13.2B total vs 1.14B activated is the intended capacity trade-off, but if optimizer/checkpoint memory outgrows hardware the design is invalid. *Mitigation:* ZeRO-3 + EP sharding of all expert and dense state; verify against the memory estimator before training.
2. **All-to-all communication cost.** 32 experts × top-2 × 16 MoE layers means dispatch/reduce every layer; on slow interconnect step time inflates even though FLOPs are flat. *Mitigation:* match EP degree to interconnect, keep cf at 1.25 (not higher), overlap comm/compute.
3. **Router collapse.** 32 experts at small scale drift toward a few active experts without a strong aux loss. *Mitigation:* aux loss 0.01, cf ≥ 1.0, optional logit jitter; monitor effective-expert count during training.
4. **Token dropping under imbalance.** Even at cf 1.25, skewed routing drops tokens, silently shrinking the effective batch. *Mitigation:* monitor overflow fraction; raise cf only if drops persist, or strengthen aux loss.
5. **Embedding term.** vocab·d = 65.5M is 5.7% of dense params here — already included; forgetting it would undercount both baselines.

Every figure above follows the `moe-architecture` formulas (`tools/moe_calculator.py`); the totals, activated = dense equality, and 11.6× ratio are the values to validate with the calculator before this doc is consumed by the training-config phase.