# MoE Architecture: 1B Dense-Equivalent on a 25 Gbps Interconnect (EP=4)

## Overview

This design replaces every FFN of a ~1B dense baseline with a 16-expert, top-2 gated mixture. The dense baseline is 24 layers, `d_model=1536`, `ffn_mult=4` (FFN hidden 6144), vocab 50,000, seq_len 2048, giving **982,843,392 (0.98B ≈ 1B)** dense parameters. The MoE variant keeps the same attention/embedding stack and swaps each FFN for a learned-softmax Top-2 router over 16 GLU experts (per-expert `3·d_model·ffn_dim = 28,311,552` params). Headline numbers: **total = 11,175,002,112 (11.18B)**, **activated = 1,662,320,640 (1.66B)**, **param ratio 11.4×** over the dense baseline. The design is built for an 8-GPU cluster (2 fixed pods × 4 GPUs) whose inter-node interconnect is capped at **25 Gbps**; the fixed expert-parallel degree is **EP = 4**, aligned exactly to one pod so that all MoE token dispatch/aggregation stays on intra-pod links and **zero bytes of token traffic ever cross the 25 Gbps link**. The only inter-pod traffic is DP gradient all-reduce (22.35 GB/step), which the batch geometry hides behind compute via comm/compute overlap.

All counts below use the `moe_calculator` formulas verbatim (`ffn_dim = ffn_mult·d_model`; per-expert GLU `= 3·d_model·ffn_dim`; attention/layer `= 4·d_model²`; embedding `= vocab·d_model`; `flops = 6·activated + 4·layers·d_model·seq_len`), so totals recompute exactly.

## Constraint satisfaction: expert-parallel degree 4 and the 25 Gbps budget

**Expert-parallel degree used: 4.** `num_experts = 16` divides EP=4 (4 experts per rank), and each EP group of 4 GPUs is placed inside one physical pod.

**Token-communication volume per step.** With micro-batch 1, seq_len 2048, grad_accum 32, DP=2 → **131,072 tokens per step**. Per MoE layer, each token dispatches top_k=2 copies of a `d_model` vector and receives the same back:

- dispatch = 131,072 × 2 × 1,536 × 2 B (bf16) = **805.3 MB/layer**
- return (combine) = **805.3 MB/layer**
- **1.61 GB per MoE layer** × 24 layers = **38.65 GB per step aggregate** (dispatch + return)
- Off-rank share with EP=4/16 (12 of 16 experts remote) = 75% → **≈ 29.0 GB/step** actually crosses rank boundaries

**How it fits 25 Gbps.** The 29.0 GB/step crosses *rank* boundaries, but because the EP group is co-located with a pod, it crosses *intra-pod* links only: ≈ 29.0 GB / 1.35 s (compute-bound step) ≈ **22 GB/s**, well within a ≥50 GB/s-class intra-pod fabric (NVLink/NVSwitch). The 25 Gbps inter-node link carries **no token traffic at all**. Its only load is the DP=2 gradient all-reduce: 11,175,002,112 params × 2 B = **22.35 GB/step**, which at the 25 Gbps = 3.125 GB/s budget takes **7.15 s/step**. The design therefore fixes the global batch so per-step compute ≥ 7.15 s: at ~40% MFU on 8 A100s (~97K tokens/s compute-bound), that means **global batch ≥ 0.69M tokens (grad_accum ≥ 170)**, overlapping the all-reduce behind the forward/backward pass and keeping the link at ≤100% utilization while hidden. Sustained throughput ≈ **97K tokens/sec**. With a too-small batch (e.g. 131K), the run becomes gradient-sync-bound at ~18K tokens/s (see Risks).

## Parameters table

`ffn_dim = 6144`; per-expert GLU `= 3·1536·6144 = 28,311,552`; attention/layer `= 4·1536² = 9,437,184`; layernorm/layer `= 2·1536 = 3,072`; embedding `= 50,000·1536 = 76,800,000`.

| Component (all 24 layers) | Dense | MoE | Activated |
| --- | --- | --- | --- |
| Attention | 226,492,416 | 226,492,416 | 226,492,416 |
| Layernorms | 73,728 | 73,728 | 73,728 |
| Expert FFNs (Dense: 1×; MoE: 16×; Activated: top-2=2×) | 679,477,248 | 10,871,635,968 | 1,358,954,496 |
| Embedding (vocab 50,000 × 1,536) | 76,800,000 | 76,800,000 | 76,800,000 |
| **Total** | **982,843,392 (0.98B)** | **11,175,002,112 (11.18B)** | **1,662,320,640 (1.66B)** |

**`num_experts = 16`**, **`top_k = 2`**. `param_ratio = 11.4×`; activated/dense = 1.69×. Router weights are negligible (`16·1536 ≈ 24.6K`/layer) and omitted, per the calculator.

## Routing

- **Strategy: learned softmax Top-2 gating**, `top_k = 2`. Top-2 is the training-quality default (knowledge blending, large gain over Top-1) and, crucially, its extra dispatch volume is **free on the constrained link** because all-to-all stays intra-pod — so there is no bandwidth reason to drop to Top-1.
- **Capacity factor = 1.0.** The aux-loss keeps load balanced so capacity 1.0 needs no padding and no slack; any capacity >1.0 would inflate dispatch/padding volume, which this design deliberately minimizes. Balanced load at capacity 1.0 = 256 tokens/expert/micro-batch (no drops, no waste).
- **Auxiliary loss = 0.01** (upper band) **plus small router-logit jitter at train time.** 16 experts with top-2 needs active balancing; jitter prevents deterministic favoritism, and the 0.01 scale stops collapse without distorting routing.

## Training implications

- **Compute.** FLOPs/token: MoE **10.28 GFLOPs** vs dense **6.20 GFLOPs** (1.66×), i.e. the top-2 MoE runs ~1.7× the 1B-dense compute budget — the standard, disclosed cost of activating two expert FFNs. At ~40% MFU on 8×A100 (2.5 PFLOPs peak) this sustains **~97K tokens/sec** once the gradient sync is hidden.
- **Memory per GPU (A100-80GB, BF16, EP=4 sharding, activation recompute):** params 6.05 GB (experts 5.44 GB = 10.87B/4, dense 0.61 GB), gradients 6.05 GB, AdamW states 36.3 GB (12 B/param), activations ~0.8–1.5 GB → **≈ 50 GB, 38% headroom** (>20% target). Without EP, the 10.87B expert params would alone be 21.7 GB/GPU.
- **Parallelization.** `DP=2 × EP=4 = 8` GPUs; EP=4 divides 16 experts. One all-to-all per MoE layer (24/step) on intra-pod links; one gradient all-reduce per step (22.35 GB) on the 25 Gbps link, hidden by `grad_accum ≥ 170` (global batch ≥ 0.69M tokens) with comm/compute overlap. Generalizes: adding pods raises DP and both compute and aggregate gradient bytes roughly linearly, so the same large-batch-overlap rule applies.

## Risks

1. **Gradient-sync-bound training** (highest risk). The fixed 7.15 s all-reduce dominates any step shorter than that; a naive 131K-token batch collapses throughput to ~18K tokens/sec. Mitigation: global batch ≥ 0.69M tokens + comm/compute overlap; consider quantized/1-bit all-reduce as a future lever.
2. **Over-parameterization.** 11.18B total params = 22.3 GB BF16 checkpoints and 36.3 GB optimizer/GPU; EP=4 sharding keeps this on-device, but an 8-expert variant (5.74B total, half the gradient bytes) is the fallback if checkpoint/comm budget tightens.
3. **Activated params exceed the 1B compute target** (1.66B active, 1.66× dense FLOPs). If strict 1B-compute parity is required, drop to Top-1: activated → 0.98B and dispatch volume halves — accept the quality loss.
4. **Router collapse / expert imbalance.** 16 experts, top-2: aux loss 0.01 + jitter; monitor effective-expert count and Gini each checkpoint.
5. **Intra-pod fabric assumption.** The design is only valid if intra-pod links are ≥ ~25 GB/s-class; if a "pod" is itself on 25 Gbps, the 29 GB/step all-to-all (~22 GB/s) exceeds budget and the design is invalid — verify the intra-pod fabric before scaling.
6. **Large global batch (0.69M) convergence.** Requires warmup/LR-schedule adjustment; and at this scale the 76.8M embedding is 7.8% of the dense baseline — it must not be dropped from the parameter math.

**Assumption noted:** cluster = 8 GPUs (2 pods × 4); per-GPU HBM 80 GB (A100-class); intra-pod fabric ≥50 GB/s (NVLink-class); inter-node 25 Gbps Ethernet, no InfiniBand; global batch 0.69M tokens chosen so the 7.15 s gradient sync is fully overlapped.