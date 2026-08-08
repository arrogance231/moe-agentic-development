Note: this environment has no shell/calculator access, so I validated every figure by hand using the exact formulas from `moe-architecture` (`tools/moe_calculator.py`). All numbers below are recomputable line-by-line from the shown math.

---

# 1B Dense-Equivalent MoE — Architecture Document

## Overview

This design replaces a ~1.14B-parameter dense decoder with a **Top-2, 64-expert Mixture-of-Experts** model. The dense baseline uses 16 layers, `d_model=2048`, `ffn_mult=4` (FFN width 8192), vocab 32000, seq_len 2048. The MoE variant keeps the attention and layer norms intact and replaces each dense FFN with 64 GLU experts, routing each token to the top-2. Headline numbers: **total params ≈ 51.87B**, **activated params ≈ 1.94B** (1.71× the dense baseline — the standard MoE "activated ≈ 1.7–1.9× dense" regime, e.g. Mixtral), **param ratio ≈ 45.5×**. Training FLOPs are 11.94 GFLOP/token vs 7.10 for dense (+68%), matching the activated-parameter ratio.

> Assumed spec (task did not supply vocab/seq_len): `vocab=32000`, `seq_len=2048`, `ffn_mult=4`. If your vocab/seq_len differ, recompute with `tools/moe_calculator.py` — the embedding and attention terms are the only ones affected.

## Parameter math (explicit)

```
ffn_dim        = ffn_mult * d_model            = 4 * 2048       = 8192
attn_per_layer = 4 * d_model^2                 = 4 * 4,194,304  = 16,777,216
expert_ffn     = 3 * d_model * ffn_dim         = 3*2048*8192    = 50,331,648
layernorm      = 2 * d_model                   = 2 * 2048       = 4,096
embedding      = vocab * d_model               = 32000 * 2048   = 65,536,000

Dense per layer     = 16,777,216 + 50,331,648 + 4,096 = 67,112,960
Dense total         = 16 * 67,112,960 + 65,536,000      = 1,139,343,360   (≈1.14B)

MoE per layer       = 16,777,216 + 4,096 + 64 * 50,331,648 = 3,238,006,784
MoE total           = 16 * 3,238,006,784 + 65,536,000      = 51,873,644,544 (≈51.87B)

Activated per layer = 16,777,216 + 4,096 + 2 * 50,331,648  = 117,444,608
Activated total     = 16 * 117,444,608 + 65,536,000        = 1,944,649,728  (≈1.94B)

param_ratio         = 51,873,644,544 / 1,139,343,360       = 45.5x
activated/dense     = 1,944,649,728 / 1,139,343,360        = 1.71x

Router weights (negligible, omitted per skill): num_experts * d_model * num_layers = 64*2048*16 ≈ 2.1M

FLOPs/token = 6 * activated_params + 4 * num_layers * d_model * seq_len
Dense  = 6*1,139,343,360 + 4*16*2048*2048 = 7,104,495,616  (≈7.10 GFLOP/token)
MoE    = 6*1,944,649,728 + 4*16*2048*2048 = 11,936,333,824 (≈11.94 GFLOP/token)
```

## Parameters

| Component | Dense | MoE | Activated |
| --- | --- | --- | --- |
| Attention (16 layers) | 268,435,456 | 268,435,456 | 268,435,456 |
| FFN / Expert FFNs (16 layers) | 805,306,368 | 51,539,607,552 | 1,610,612,736 |
| LayerNorms (16 layers) | 65,536 | 65,536 | 65,536 |
| Embedding | 65,536,000 | 65,536,000 | 65,536,000 |
| **Total parameters** | **1,139,343,360** | **51,873,644,544** | **1,944,649,728** |
| **num_experts** | — | **64** | — |
| **top_k** | — | **2** | — |

## Routing choice

- **Strategy: Top-2 learned routing** (softmax router over 64 experts, dispatch to the 2 highest logits). Top-2 is the default for training-quality-focused work: it lets tokens blend two expert representations and delivers a large quality gain over Top-1 at modest all-to-all cost. Learned/soft routing is rejected — it is harder to load-balance and unstable to train, with no need here.
- **top_k = 2** — standard training default; the 1.71× activated/dense ratio (vs ~0.85× for Top-1) is the accepted price for quality.
- **num_experts = 64** — squarely in the 16–64 production band, and 64 divides evenly across 8 GPUs (EP=8 → 8 experts/rank), keeping all-to-all tractable.
- **Capacity factor = 1.25** — absorbs token-routing imbalance without much wasted compute; 1.0 would drop tokens under skew, 1.25 leaves headroom.
- **Aux loss = 0.01** — upper end of the 0.001–0.01 band, deliberately set to guard against router collapse at a 64-expert scale; a weaker value risks a few experts swallowing all tokens.

## Training implications

- **Compute**: 11.94 GFLOP/token vs 7.10 dense = **1.68× dense FLOPs**, tracking the 1.71× activated-parameter ratio. Expert FFNs dominate (activated expert FFNs are 2× the dense FFN).
- **Memory**: 51.87B total params vs 1.94B activated. In BF16 the full weights are ~104 GB — **expert parallelism is mandatory**. With EP=8 on 8 GPUs each rank holds 8 experts (≈1.6B expert params/rank), plus its attention share; the dense/attention parts shard under DP/TP. Checkpointing (64×FFN activations) is required; micro-batch must keep ≥8–64 tokens/expert/GPU.
- **Parallelization**: DP×EP = 8×8 (EP divides 64). EP triggers **one all-to-all per MoE layer = 16 per step**, dispatching top-k=2 × tokens × capacity 1.25. Keep capacity factor consistent at 1.25 across config; a mismatch silently changes the effective batch via dropped tokens.

## Risks

1. **Over-parameterization without compute gain** — total params (51.9B) are ~45× dense while activated/FLOPs are ~1.7×. Without EP sharding the model cannot fit; the design is only valid under the EP=8 layout above.
2. **All-to-all communication** — 64 experts × top-2 × capacity 1.25 raises dispatch volume vs a smaller expert count; 16 all-to-alls per step must overlap with compute or EP=8 step time suffers.
3. **Routing collapse** — mitigated by aux loss 0.01 + capacity 1.25 so under-loaded experts keep tokens and gradients; verify with `router_distribution.py` (effective experts ≥ 0.5×64).
4. **Token drops at capacity 1.25** under severe imbalance — dropping corrupts the effective batch; raise capacity only if drops are observed, or strengthen the aux loss first.
5. **Embedding term** — 65.5M embedding is ~5.7% of the dense total; it is not what makes this model big, but forgetting it anywhere in the comparison makes the 1.14B dense-equivalent claim wrong.

---

**Config summary**: `num_layers=16, d_model=2048, ffn_mult=4, vocab=32000, seq_len=2048, num_experts=64, top_k=2, routing=top-2, capacity_factor=1.25, aux_loss=0.01`. Hand-verified against the skill's formulas; re-run `tools/moe_calculator.py` with these inputs to confirm the totals match.