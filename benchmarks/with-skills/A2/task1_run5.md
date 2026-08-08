# 1B Dense-Equivalent MoE — Architecture Document

**Assumptions (spec was open-ended, so fixed explicitly):** `num_layers=24`, `d_model=1536`, `ffn_mult=4` (`ffn_dim=6144`), `vocab=32000`, `seq_len=2048`. Dense baseline lands at ~0.96B. All counts follow the `tools/moe_calculator.py` formulas exactly.

## Overview

The dense baseline is a 24-layer, `d_model=1536` transformer with GLU FFNs at **955M parameters (0.96B)**, ~6.03 GFLOP/token at 2048-token sequences. The MoE replaces each dense FFN with **16 experts** (GLU), routing each token through the **top-2** via a learned router. Total parameters grow to **11.15B** (11.7× the dense baseline) while **activated parameters stay at 1.63B** (~1.7× the dense compute). This is the standard "1B dense-equivalent" pattern: base architecture sized as 1B, expert count inside the typical production range, activated-compute penalty kept modest.

## Parameters

Math (each figure recomputed from the skill's formulas):

```text
ffn_dim        = 4 × 1536            = 6144
attention/layer= 4·d_model²          = 9,437,184
expert FFN     = 3·d_model·ffn_dim   = 28,311,552   (GLU: 2 in, 1 out)
layernorms     = 2·d_model           = 3,072
embedding      = 32000 × 1536        = 49,152,000

dense  = 24·(9,437,184 + 28,311,552 + 3,072) + 49,152,000 = 955,195,392
moe    = 24·(9,437,184 + 3,072 + 16×28,311,552) + 49,152,000 = 11,147,354,112
active = 24·(9,437,184 + 3,072 + 2×28,311,552) + 49,152,000 = 1,634,672,640
param_ratio = 11,147,354,112 / 955,195,392 = 11.7×
```

| Component | Dense | MoE | Activated |
| --- | --- | --- | --- |
| Attention (24 layers) | 226,492,416 | 226,492,416 | 226,492,416 |
| Expert FFNs (24 layers) | 679,477,248 | 10,871,635,968 | 1,358,954,496 |
| LayerNorms (24 layers) | 73,728 | 73,728 | 73,728 |
| Embedding | 49,152,000 | 49,152,000 | 49,152,000 |
| **Total** | **955,195,392** | **11,147,354,112** | **1,634,672,640** |
| **num_experts** | — | **16** | — |
| **top_k** | — | — | **2** |

Router weights (`num_experts·d_model = 24,576`/layer) are negligible and omitted per the calculator.

## Routing choice

- **Strategy: Top-2 (learned).** The training-focused default — blends two experts' knowledge and is the established quality sweet spot over Top-1 at a modest ~2× dispatch cost. Learned gating over soft/heuristic routing.
- **top_k = 2:** matches the training default; keeps activated compute at 1.7× dense.
- **Capacity factor = 1.25:** absorbs token-routing imbalance without meaningful wasted compute; 1.0 would drop tokens under imbalance at this scale.
- **Auxiliary loss = 0.01** (`router_aux_loss_coef`): top of the 0.001–0.01 band — with 16 experts on a 1B base, routing entropy is small enough that 0.01 keeps all experts loaded without distorting the router objective.

## Training implications

- **Compute:** 10.11 GFLOP/token vs 6.03 dense (1.68×), so per-token cost stays under 2× despite 11.7× parameters.
  ```text
  flops_moe   = 6×1,634,672,640 + 4·24·1536·2048 = 10,110,025,728  (10.11 GFLOP/token)
  flops_dense = 6×955,195,392   + 4·24·1536·2048 = 6,033,162,240   (6.03 GFLOP/token)
  ```
- **Memory:** optimizer states (AdamW, 12 B/param) dominate. On 8 GPUs with EP=8 (2 experts/rank), per-rank params ≈ 1.63B → ~3.3 GB bf16 params + 3.3 GB grads + ~19.6 GB optimizer ≈ ~26 GB + activations — fits an 80 GB H100 with headroom; on 40 GB GPUs add gradient checkpointing or ZeRO-shard the dense parts.
- **Parallelization:** EP=8 divides `num_experts=16` cleanly; `DP×TP×PP×EP = 8` GPUs. One all-to-all dispatch/receive per MoE layer (24 total) — the dominant comm cost, bounded by top-2 × capacity 1.25.
- **Batch geometry:** aim for 8–64 tokens per expert per GPU per micro-batch; capacity factor must stay 1.25 across config and architecture doc (mismatch silently drops tokens).

## Risks

- **Over-parameterization:** 11.1B total vs 1.63B activated — checkpoint/memory burden grows ~12× while compute stays flat. *Mitigation:* EP=8 sharding plus ZeRO for the dense optimizer states; if HBM is tight, drop to 8 experts (→ ~5.7B total, same activated).
- **All-to-all cost:** top-2 × 16 experts × capacity 1.25 over 24 layers can bottleneck on slow interconnect. *Mitigation:* keep EP=8 within a fast node (NVLink); raise EP only if comm-bound.
- **Routing collapse:** weak aux loss leaves experts idle. *Mitigation:* aux loss 0.01; verify effective experts ≥ 8 (0.5×n) early in training.
- **Token dropping at capacity 1.25:** imbalance can still overflow buffers. *Mitigation:* monitor overflow fraction; raise to 1.5 only if drops appear, else leave at 1.25.
- **Embedding share at small scale:** the 49M embedding is 5% of the dense baseline — already included in every total; keep vocab fixed when comparing against the baseline.

Numbers are exact per the `moe_calculator.py` formulas; validate by re-running the calculator before training.