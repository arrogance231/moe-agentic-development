I'll design this MoE architecture under the fixed constraints. The skill's parameter/FLOP formulas are used exactly as given (dense baseline = 14×2048 model ≈ 1B, hybrid dense/MoE layers to control dispatch volume).

---

# Architecture Document: 1B-Dense-Equivalent MoE on a 25 Gbps Interconnect (EP fixed at 4)

## Overview

The dense-equivalent baseline is a 1,042,507,776-parameter (~1.04B) causal LM: 14 layers, `d_model = 2048`, `ffn_mult = 4` (`ffn_dim = 8192`), vocab 50,257, `seq_len = 2048`. The MoE variant keeps all 14 layers but replaces the FFN in **4 of 14 layers** with a 16-expert, **top-2** mixture (layers 2, 5, 8, 11); the other 10 layers keep a standard dense FFN. This hybrid is the central response to the interconnect constraint: it captures ~3.9× parameter growth while cutting expert-dispatch traffic to 4/14 ≈ 29% of a full-MoE design.

**Headline numbers:** total params **4,062,406,656 (≈4.06B)**, activated params **1,243,834,368 (≈1.24B)**, param ratio **3.90×** over dense, and FLOPs/token only **1.19×** dense (7,697,887,232 vs 6,489,927,680). The design is **token-communication-bound, not compute-bound**: at the target ~40k tokens/s the per-step expert all-to-all volume is **8.59 GB/step = 21.0 Gbps**, under the 25 Gbps budget with ~16% headroom.

**Constraints honored:**
- **Expert-parallel degree = 4** (fixed; `EP` divides `num_experts` = 16, so 4 experts per rank).
- **Per-step token-communication volume = 8,589,934,592 bytes (8.59 GB)** — derivation in *Training implications*.
- **Within 25 Gbps:** 8.59 GB amortized over a 3.28 s step = 2.62 GB/s = 21.0 Gbps ≤ 25 Gbps.

## Parameters table

Formulas follow the `moe-architecture` skill calculator definitions (`tools/moe_calculator.py`): attention ≈ `4·d² + 2·d`, dense FFN = `3·d·ffn`, per-expert FFN (GLU) = `3·d·ffn`, embedding = `vocab·d`.

| Component | Dense (all 14 layers) | MoE (4 of 14 layers) | Activated (top-2) |
| --- | --- | --- | --- |
| Attention + LN, all 14 layers | 234,938,368 | 234,938,368 | 234,938,368 |
| Dense FFN, 10 layers | 503,316,480 | 503,316,480 | 503,316,480 |
| Expert FFNs, 4 layers × 16 experts | — | 3,221,225,472 | 402,653,184 |
| Embedding (vocab 50,257 × 2048) | 102,926,336 | 102,926,336 | 102,926,336 |
| **Total** | **1,042,507,776** | **4,062,406,656** | **1,243,834,368** |

- **`num_experts` = 16**, **`top_k` = 2**, capacity factor 1.25, aux loss 0.01.
- Param ratio (MoE/dense) = **3.897×**; total/activated = **3.27×**.
- Per-layer figures: attention+LN = 16,781,312; dense FFN = 50,331,648; expert FFN/layer = 805,306,368; FLOPs/token = 7.70e9 (MoE) vs 6.49e9 (dense).

## Routing

- **Strategy: learned Top-2.** Top-2 is the standard quality/latency sweet spot for training; with only 4 MoE layers the per-token dispatch cost is affordable and Top-1's knowledge-blending loss is unnecessary here.
- **top_k = 2.** Keeps quality at 1.19× dense compute; also the control point for communication volume (a Top-1 fallback halves dispatch bytes, see Risks).
- **Capacity factor = 1.25.** Absorbs token-routing imbalance (which would otherwise drop tokens and inflate all-to-all bursts) at modest padding cost.
- **Aux loss = 0.01** (upper band). The small EP degree and 25 Gbps budget leave no room for routing collapse or imbalance-driven dispatch spikes; a strong load-balancing loss keeps per-expert load uniform and the all-to-all volume predictable.

## Training implications

**Compute:** 7.70 GFLOPs/token vs 6.49 dense = **1.19×**. On 16 GPUs (4 pods × 4; A100-class bf16 ≈ 312 TFLOPS each), raw compute could sustain ~260k tokens/s at 40% MFU — the interconnect, not compute, is the pacing resource.

**Parallelism:** `DP=4 × TP=1 × PP=1 × EP=4 = 16` GPUs; `EP=4` divides `num_experts=16` (4 experts/rank). `TP=1` and `PP=1` by design: tensor/pipeline parallelism exchange traffic every op/layer and would thrash a 25 Gbps fabric. One all-to-all per MoE layer → **4 all-to-alls per step**.

**Batch geometry:** micro-batch 8, `seq_len` 2048, DP 4, grad-accum 2 → global batch **131,072 tokens/step**. Tokens per expert per GPU per micro-batch = `8·2048·2/(16/4)` = 8,192, far above the 8–64 utilization floor.

**Token-communication volume per step (expert all-to-all, bf16 dispatch):**

```
V = 2 (dispatch + receive) × 4 MoE layers × top_k(2) × B(131,072) × d_model(2048) × 2 B
  = 2 × 4 × 2 × 131,072 × 2,048 × 2 = 8,589,934,592 B ≈ 8.59 GB/step
```

**25 Gbps budget check:** budget = 25 Gbps = 3.125e9 B/s. The volume fits iff step time ≥ 8.59e9/3.125e9 = **2.75 s**. At the target 40k tokens/s, a 131,072-token step takes 3.28 s, so the sustained rate is **8.59 GB / 3.28 s = 21.0 Gbps ≤ 25 Gbps (84% utilization, ~16% headroom)**. Equivalently, the interconnect caps this design at 47,683 tokens/s (`3.125e9 / 65,536 B-per-token`), and we deliberately run below that. Gradient accumulation lengthens the per-step window at fixed volume, which is exactly the lever used here.

**Memory (per GPU):** full model states ≈ 16 B/param (bf16 params+g grads + fp32 AdamW master/mom/var) ≈ 65.0 GB, ZeRO-2-shattered over 16 ranks ≈ 4.1 GB; activations ≈ 9.4 GB (14·8·2048·2048·20/TP), ~4.7 GB with recomputation; total ≈ 9.4 GB/GPU ≪ 40 GB — comfortable headroom even at EP=4.

## Risks

- **Interconnect-bound throughput (primary).** At 40k tokens/s the design runs at ~6% MFU; compute headroom is ~6×. This is the unavoidable cost of dispatching tokens over 25 Gbps — no published recipe assumes it because it is inefficient by construction. Mitigation: keep `L_MoE` small (4 layers), run at ≤ 40k tokens/s, and overlap all-to-all with FFN compute.
- **Budget sensitivity.** 21.0/25 Gbps leaves only ~16% margin; an imbalance burst or larger batch pushes past budget. Mitigations: capacity factor 1.25, aux loss 0.01, and, if needed, Top-1 (halves dispatch), fewer MoE layers (2 layers → 10.5 Gbps), or LAG-bonding two 25 Gbps links.
- **Low dispatch efficiency / padding waste.** Top-2 × capacity 1.25 pads FFN batches; acceptable at this scale, since expert FFN compute is only ~12% of total FLOPs.
- **Over-parameterization without compute gain.** 4.06B total vs 1.24B activated raises checkpoint/memory cost for flat FLOPs. Mitigation: ZeRO-2 sharding keeps it to ~4.1 GB/GPU; checkpoints in bf16.
- **Router collapse/imbalance.** A collapsed router concentrates dispatch on few experts, spiking all-to-all on a few links. Mitigation: aux loss 0.01 + router-logit jitter; monitor per-expert counts (the `moe-debugging` `router_distribution` checks).
- **Capacity-factor mismatch with token drops.** If imbalance drops tokens under capacity 1.25, effective batch shrinks and dispatch bursts — track overflow fraction and raise CF toward 1.25 if drops appear.

---

All parameter counts above are hand-derived from the `moe-architecture` skill formulas (the `tools/moe_calculator.py` equivalents); totals and the 3.90× ratio should be confirmed by running the calculator in the full skill environment, which is not accessible from this harness directory.