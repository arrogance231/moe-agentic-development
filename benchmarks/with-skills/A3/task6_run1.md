# MoE Architecture: 1B Dense-Equivalent on a 25 Gbps Interconnect, Fixed EP=4

## Overview

This design targets a 1B dense-equivalent MoE trained on a cluster whose inter-node interconnect is capped at 25 Gbps and whose topology groups GPUs into fixed pods of 4. The expert-parallel degree is **fixed at 4** (`EP=4`) — one expert-parallel group maps one-to-one onto one 4-GPU pod. The dense baseline is a 0.955 B parameter decoder (24 layers, `d_model=1536`, `ffn_mult=4`). The MoE replaces every FFN with a 16-expert, **top-2** mixture, giving:

- **Total parameters: 11,147,354,112 (≈ 11.15 B)** — 16 experts, top-2
- **Activated parameters: 1,634,672,640 (≈ 1.63 B)**
- **Param ratio vs dense: 11.67×**
- FLOPs/token: 10.11 GFLOP vs 6.03 GFLOP dense (1.68×)

The headline constraint decision: because `EP=4` is pinned to the pod topology, **all token dispatch (all-to-all) stays inside the pod's fast local fabric and never crosses the 25 Gbps inter-node link**. The inter-node budget carries only dense-gradient all-reduce, which is ~1.1 GB/step — about 0.15–0.9 Gbps, i.e. **28×–170× headroom** under the 25 Gbps cap.

## Parameters table

All figures computed with the skill's parameter formulas (`tools/moe_calculator.py`): `ffn_dim = 4×1536 = 6144`, per-expert GLU FFN = `3·d·f = 28,311,552`, attention/layer = `4·d² = 9,437,184`, layernorms/layer = `2·d = 3,072`, embedding = `32,000×1536 = 49,152,000`, `num_layers = 24`.

| Component | Dense | MoE | Activated |
| --- | --- | --- | --- |
| Attention (all layers) | 226,492,416 | 226,492,416 | 226,492,416 |
| Expert FFNs (all layers) | 679,477,248 | 10,871,635,968 | 1,358,954,496 |
| Layernorms (all layers) | 73,728 | 73,728 | 73,728 |
| Embedding | 49,152,000 | 49,152,000 | 49,152,000 |
| **Total** | **955,195,392** | **11,147,354,112** | **1,634,672,640** |

`num_experts = 16` (in the 16–64 production band; divisible by the fixed `EP=4`, so 4 experts per rank). `top_k = 2`. `param_ratio = 11.67×`.

## Routing

| Setting | Value | Justification |
| --- | --- | --- |
| Strategy | Learned Top-2 (token-choice) | Training default; keeps quality headroom over Top-1 |
| top_k | 2 | The normal penalty of +1 expert is dispatch/all-to-all cost; here that cost is intra-pod, so Top-2 is affordable |
| Capacity factor | 1.25 | Absorbs routing imbalance without dropping tokens |
| Aux loss | 0.01 | Upper end of 0.001–0.01: prevents router collapse (critical, since capacity padding also inflates dispatch volume) |

I did **not** drop to Top-1 to save communication: with `EP=4` colocated inside a pod, the all-to-all Top-2 adds is free of the inter-node budget (see below). Top-1 remains the documented fallback if pod placement is later violated.

## Training implications

**Parallelism (16 GPUs = 4 pods × 4).** `DP=4 × TP=1 × PP=1 × EP=4 = 16`; `EP=4` divides `num_experts=16`. No TP/PP: they would add inter-node all-reduce/p2p traffic on the 25 Gbps fabric for no benefit. `EP=4` = fixed, and each EP group is placed entirely within one pod.

**Batch geometry.** `micro_batch=8`, `seq_len=2048` → 16,384 tokens/micro-batch/rank; `grad_accum=16`, `DP=4` → **global batch = 1,048,576 tokens/step (≈1M)**. Tokens per expert per GPU per micro-batch = `8×2048×2 / (16/4) = 8,192`, far above the 8–64 utilization floor.

**Compute.** 10.11 GFLOP/token (1.68× dense). At ~1M tokens/step in ~60 s → ~17k tokens/sec aggregate → MFU ≈ 3–4% on 16×A100: honestly low, and expected — this cluster is topology/bandwidth-bound, not compute-bound.

**Memory per GPU (80 GB HBM, BF16, gradient checkpointing).** Expert params 2.72 B/rank (5.44 GB) + ZeRO-sharded dense 0.07 B (0.14 GB); grads 5.58 GB; AdamW-mixed optimizer 33.5 GB; activations ≈6.0 GB (recomputed, halved). Total ≈ **57 GB → ~28% headroom** (≥20% target).

### Interconnect budget — the load-bearing number

Per MoE layer, per step, all-to-all (dispatch + receive) volume:

```
2 (dispatch+receive) × top_k(2) × 1,048,576 tokens × d_model(1536) × 2 B (bf16)
  = 12,884,901,888 B ≈ 12.88 GB/layer
× 24 MoE layers = 309,237,645,312 B ≈ 309.2 GB/step
```

25 Gbps = 3.125 GB/s. **0 GB of this 309.2 GB/step crosses the inter-node link**: the design places each `EP=4` group inside one 4-GPU pod, so every all-to-all (24 per step) runs on the pod-local fabric (NVLink-class, ~10–30 GB/s at 30–60 s steps — fits). The inter-node link carries only the DP gradient all-reduce of dense parameters (attention + layernorms + embedding = 275,718,144 params → 551 MB of BF16 grads, ≈1.10 GB/step effective across the DP=4 ring): **0.15 Gbps at 60 s steps, 0.88 Gbps at 10 s steps — 28×–170× under the 25 Gbps cap.**

*Sensitivity:* if the EP group were forced to span pods, 309.2 GB/step over 3.125 GB/s would force step time ≥ 99 s (≈10.6k tok/s ceiling, ~2% MFU) — so colocation is load-bearing, not cosmetic. Documented fallbacks if it breaks: Top-1 (halves to 154.6 GB), MoE-every-other-layer, or accept the ceiling. Padding at cf=1.25 can scale the *intra-pod* volume toward ~1.25× — still on NVLink, never on the 25 Gbps fabric.

**Checkpointing.** BF16 safetensors, 11.15 B params ≈ 22.3 GB/checkpoint → ~7 s transfer over 25 Gbps; negligible at infrequent cadence, but stage to local NVMe to avoid blocking.

## Risks

1. **Inter-node oversubscription if EP placement is not enforced.** All headroom depends on `EP=4` living inside one pod. Mitigation: pin rank order to pods at launch, verify topology, and keep the Top-1 / fewer-MoE-layers fallbacks.
2. **Router collapse.** With aux loss too weak, few experts carry traffic and padding inflates dispatch. Mitigation: aux 0.01, cf ≥ 1.0, monitor effective experts.
3. **Padding waste at cf=1.25.** Inflates intra-pod dispatch and compute. Mitigation: tighten cf to 1.0 once balance is confirmed; sequence packing.
4. **Over-parameterization (11.15 B total vs 0.96 B dense-equivalent).** Memory/checkpoint cost. Mitigation: per-GPU fits with 28% headroom; BF16; gradient checkpointing.
5. **Stragglers across the 4 EP ranks.** Skew leaves pod capacity idle. Mitigation: router jitter, per-expert load monitoring.

---

**Constraint status:** expert-parallel degree = **4** (fixed, one pod per EP group); token-communication volume per step = **309.2 GB** (ideal) of all-to-all, **all intra-pod**; inter-node load = **≤ ~1.1 GB/step of dense-gradient traffic ≈ 0.9 Gbps max**, kept inside the **25 Gbps** budget by pod-colocating the fixed EP degree so no all-to-all crosses the inter-node fabric.