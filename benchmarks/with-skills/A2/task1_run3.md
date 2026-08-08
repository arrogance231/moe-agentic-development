# 1B Dense-Equivalent MoE — Architecture Document

## Overview

This design replaces a ~1.005B dense decoder baseline with a Mixture-of-Experts variant that keeps **activated parameters identical to the dense baseline** (~1.005B) — so it is a true *compute-equivalent* swap — while expanding total parameters to ~5.94B for representational capacity. The MoE replaces each dense FFN with **16 experts (top-2 active)**, each expert a GLU FFN of half the dense FFN width. Routing is **learned Top-2** with capacity factor **1.25** and auxiliary load-balancing loss **0.01**. Activated params = dense params exactly (2 experts × half-width FFN = 1 dense-width FFN), so training FLOPs/token are unchanged from the dense baseline at ~6.27 GFLOPs/token.

**Headline numbers:** total params **5.94B**, activated params **1.005B**, param ratio **5.91×**, top-k **2**, num_experts **16**.

## Parameters

Dense baseline: `num_layers=14`, `d_model=2048`, `ffn_mult=4` → `ffn_dim=8192`, `vocab=32,000`, `seq_len=2,048`.
MoE: `num_experts=16`, `top_k=2`, per-expert `ffn_dim=4096`.

**Math (per the `moe_calculator.py` formulas):**

- Attention per layer: `4·d_model² = 4·2048² = 16,777,216`
- Dense FFN (GLU) per layer: `3·d_model·ffn_dim = 3·2048·8192 = 50,331,648`
- Per-expert FFN (GLU): `3·d_model·ffn_dim_expert = 3·2048·4096 = 25,165,824`
- Layernorms per layer: `2·d_model = 4,096`
- Embedding: `vocab·d_model = 32,000·2,048 = 65,536,000`

| Component | Dense | MoE | Activated |
| --- | --- | --- | --- |
| Attention (all layers, ×14) | 234,881,024 | 234,881,024 | 234,881,024 |
| Expert FFNs (all layers) | 704,643,072 | 5,637,144,576 | 704,643,072 |
| Layernorms (all layers, ×14) | 57,344 | 57,344 | 57,344 |
| Embedding | 65,536,000 | 65,536,000 | 65,536,000 |
| **Total** | **1,005,117,440** | **5,937,618,944** | **1,005,117,440** |

- Dense: `14·(16,777,216 + 50,331,648 + 4,096) + 65,536,000 = 1,005,117,440`
- MoE: `14·(16,777,216 + 16·25,165,824 + 4,096) + 65,536,000 = 5,937,618,944`
- Activated: `14·(16,777,216 + 2·25,165,824 + 4,096) + 65,536,000 = 1,005,117,440`
- **Total params:** `5,937,618,944` (~5.94B) · **Activated:** `1,005,117,440` (~1.005B) · **num_experts:** `16` · **top_k:** `2` · **param_ratio:** `5.94×` (5,937,618,944 / 1,005,117,440 ≈ 5.91)

**FLOPs/token** (dense and MoE identical by construction):
`6·activated + 4·num_layers·d_model·seq_len = 6·1,005,117,440 + 4·14·2048·2048 = 6,030,704,640 + 234,881,024 = 6,265,585,664` (~6.27 GFLOPs/token).

## Routing choice

| Setting | Value | Justification |
| --- | --- | --- |
| Strategy | **Top-2 (learned)** | Default for training-quality work; the sweet spot over Top-1 at modest all-to-all cost. |
| top_k | **2** | With half-width expert FFNs, top-2 exactly reproduces the dense FFN compute — the property that keeps activated params at 1B. |
| Capacity factor | **1.25** | Absorbs token-routing imbalance without wasting much compute; 1.0 risks dropped tokens under skew. |
| Aux loss | **0.01** | Upper end of the 0.001–0.01 band; at 16 experts on a ~1B model, imbalance/collapse risk outweighs routing distortion, so bias toward load balancing. |

## Training implications

- **Compute:** ~6.27 GFLOPs/token — identical to the dense baseline. MoE adds no FLOPs; it trades memory for capacity.
- **Memory:** total params 5.94B vs activated 1.0B. Checkpoint/optimizer cost grows ~5.9× over dense; activations are unchanged (~dense), since only FFN width per token is the same.
- **Parallelization (8 GPUs):** `DP=1 × TP=1 × PP=1 × EP=8` — product = 8, and EP=8 divides `num_experts=16` (2 experts per rank per layer). One all-to-all per MoE layer (14 total). Dense blocks (attention + embedding, ~300M) replicate across ranks; experts shard by EP.
- **Training-time guardrails:** jitter on router logits and the aux loss prevent early specialization; gradient clipping 1.0 and BF16 master weights guard stability.

## Risks

- **Routing collapse / imbalance.** Weak aux loss starves under-used experts. *Mitigation:* aux loss 0.01, router-logit jitter, capacity factor ≥1.0.
- **All-to-all overhead.** 16 experts × top-2 doubles dispatch vs Top-1. *Mitigation:* EP=8 (2 experts/rank) keeps volume modest; EP must divide 16.
- **Over-parameterization.** 5.94B total params (5.9× dense) raises checkpoint and sharded-optimizer memory. *Mitigation:* expert + optimizer sharding via EP/ZeRO; half-width experts keep activated math at the 1B target.
- **Token drops.** Capacity 1.25 is safe for mild imbalance but drops tokens if routing skews hard. *Mitigation:* monitor overflow; raise to a higher cap or strengthen aux loss before raising capacity (which wastes compute).

All figures follow the `moe_calculator.py` formulas; note I could not execute the script here (no shell in this environment), so counts were computed by hand from those formulas and should be re-validated with the calculator.