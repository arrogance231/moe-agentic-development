I don't have shell access in this session, so I'll compute the parameter math by hand exactly per `tools/moe_calculator.py`'s formulas (which the skill specifies inline) and present it explicitly.
# 1B Dense-Equivalent MoE — Architecture Document

## Overview

The baseline is a dense transformer of **1.005B parameters** (`num_layers=14`, `d_model=2048`, `ffn_mult=4`, `vocab=32000`, `seq_len=2048`). The MoE variant replaces every dense FFN with a router-gated Mixture-of-Experts layer of **64 experts, top-2 active** (Top-2 routing), giving **45.40B total parameters** at **1.71B activated parameters** per token — a **45.2× total param growth** for only **1.70×** the dense-activated count, so FLOPs grow to ~1.67× the dense baseline while the token-routing capacity scales the model's memory footprint by the expert count.

## Parameters

Configuration: `num_layers=14`, `d_model=2048`, `ffn_mult=4`, `vocab=32000`, `seq_len=2048`, `num_experts=64`, `top_k=2`.

**Parameter math (explicit):**

```text
d_model     = 2048
ffn_dim     = ffn_mult * d_model = 4 * 2048 = 8192
attention   = 4 * d_model² = 4 * 4,194,304 = 16,777,216 / layer
expert FFN  = 3 * d_model * ffn_dim = 3 * 2048 * 8192 = 50,331,648 / expert / layer
layernorms  = 2 * d_model = 4,096 / layer
embedding   = vocab * d_model = 32,000 * 2048 = 65,536,000

dense_params  = 14*(4*2048² + 3*2048*8192 + 2*2048) + 32,000*2048
              = 14*(16,777,216 + 50,331,648 + 4,096) + 65,536,000
              = 14 * 67,112,960 + 65,536,000
              = 939,581,440 + 65,536,000
              = 1,005,117,440                        (~1.005B)

moe_params    = 14*(4*2048² + 2*2048 + 64 * 3*2048*8192) + 32,000*2048
              = 14*(16,777,216 + 4,096 + 64 * 50,331,648) + 65,536,000
              = 14*(16,777,216 + 4,096 + 3,221,225,472) + 65,536,000
              = 14 * 3,238,006,784 + 65,536,000
              = 45,332,094,976 + 65,536,000
              = 45,397,630,976                        (~45.40B)

activated_params = 14*(4*2048² + 2*2048 + 2 * 3*2048*8192) + 32,000*2048
              = 14*(16,777,216 + 4,096 + 2 * 50,331,648) + 65,536,000
              = 14*(16,777,216 + 4,096 + 100,663,296) + 65,536,000
              = 14 * 117,444,608 + 65,536,000
              = 1,644,224,512 + 65,536,000
              = 1,709,760,512                         (~1.71B)

param_ratio = 45,397,630,976 / 1,005,117,440 = 45.2
```

| Component (all layers) | Dense | MoE | Activated |
| --- | --- | --- | --- |
| Attention (14 × 4·d²) | 234,881,024 | 234,881,024 | 234,881,024 |
| Expert FFNs (14 × 64 × 3·d·ffn) | 704,643,072¹ | 45,097,156,608 | 1,409,286,144² |
| Layernorms (14 × 2·d) | 57,344 | 57,344 | 57,344 |
| Embedding (vocab × d) | 65,536,000 | 65,536,000 | 65,536,000 |
| **Total** | **1,005,117,440** | **45,397,630,976** | **1,709,760,512** |

¹ Dense FFN total for reference. ² Only `top_k=2` expert FFNs are active per token.

**Total parameters: 45,397,630,976 (≈45.4B).** `num_experts = 64`. `top_k = 2`.

## Routing choice

- **Strategy: Top-2** — the standard default for training-quality-focused runs. Two active experts per token keep knowledge blending (quality edge over Top-1) at a modest all-to-all cost (2× dispatch vs Top-1), which is the right trade for a training-driven design.
- **Capacity factor: 1.25** — absorbs token-routing imbalance across 64 experts without dropping tokens early in training; keeps load-balancing signal clean.
- **Auxiliary loss: 0.01** — upper end of the 0.001–0.01 band. With 64 experts the router collapse risk is high; the stronger coefficient is needed to hold the effective expert count near the full 64.
- Add train-time jitter to router logits per the skill's imbalance countermeasures.

## Training implications

- **Compute:** FLOPs/token = `6 * activated + 4 * num_layers * d_model * seq_len`:
  - Dense: `6 * 1,005,117,440 + 4 * 14 * 2048 * 2048 = 6.27B`
  - MoE: `6 * 1,709,760,512 + 4 * 14 * 2048 * 2048 = 10.49B` → **1.67× dense**.
  Compute scales with the *activated* (1.71B) count, not the 45.4B total.
- **Memory:** total params in BF16 ≈ 90.8 GB — far above the activated footprint, so this design **only fits via expert parallelism (EP)**. On 8 GPUs, EP=8 puts 8 experts/rank (≈5.7B params ≈ 11.3 GB BF16 per rank) plus the dense attention/embedding layers; optimizer states require ZeRO/DeepSpeed sharding over the DP group.
- **Parallelization:** `EP=8` divides `num_experts=64` cleanly; **one all-to-all per MoE layer** (14 per step), with dispatch volume `top_k × capacity × tokens`. Attention is replicated across ranks; no TP required at this scale.
- **Checkpointing/context:** at `seq_len=2048` with the 45B-parameter footprint, gradient checkpointing is advisable to keep activations (and capacity-factor buffers at cf=1.25) within HBM.

## Risks

- **Over-parameterization without compute gain** — the 45.4B total buys 1.67× FLOPs but a ~45× checkpoint/memory footprint; invalid on any hardware that can't shard the experts. **Mitigation:** EP=8+ on multi-GPU, expert offload if HBM-bound.
- **All-to-all cost at 64 experts** — 64 experts raise dispatch/summing traffic; past the device count this dominates step time even with flat FLOPs. **Mitigation:** keep EP degree ≥ 8, capacity factor 1.25 to bound volume, and monitor step-time share.
- **Routing collapse** — 64 experts with top-2 is collapse-prone; the router can degenerate to a few experts. **Mitigation:** aux loss 0.01 + logit jitter; watch effective expert count.
- **Token drops from capacity 1.25** — if load becomes skewed, drops distort the effective batch. **Mitigation:** strengthen aux loss first, lower cf only after balance is confirmed.
- **Embedding term at 1B scale** — 65.5M (6.5% of dense) is material at this size; it is included above so totals are not undercounted.

## Evaluation cross-check

- Parameter math recomputed exactly per the `moe_calculator.py` formulas inline (no shell available in this session); totals and the 45.2× ratio follow from the arithmetic above.
- `num_experts=64` and `top_k=2` match the skill's shared 7B-MoE / ~1B dense-equivalent reference config (64 experts, top-2, 8 GPUs).
- Routing, capacity factor, and aux loss are each defended against a training-quality use case.
Done. Design: **14 layers, d_model=2048, ffn_mult=4, vocab=32000** → 1.005B dense-equivalent; MoE with **64 experts, top-2**, capacity factor **1.25**, aux loss **0.01**, Top-2 routing → **45.4B total / 1.71B activated** (45.2× ratio), 1.67× dense FLOPs. Numbers computed exactly per the `moe_calculator.py` formulas; note I had no shell access here, so run `tools/moe_calculator.py` on this config to confirm.