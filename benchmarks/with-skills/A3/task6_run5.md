I don't have shell/file access in this environment, so I applied the `moe-architecture` skill's formulas directly and worked every number by hand (all totals below are arithmetic-exact per the calculator's formulas in the skill). Here is the architecture document.

---

# MoE Architecture: 1B Dense-Equivalent under a 25 Gbps Interconnect Cap

## Overview

The dense-equivalent baseline is a 24-layer, d_model=1536, GLU-FFN (ffn_mult=4, ffn_dim=6144), vocab=32000 transformer — **955,195,392 parameters (≈0.96B, "1B dense-equivalent")**. The MoE variant keeps the same 24 layers, width, and embedding, replacing every dense FFN with **8 experts, top-k=2** (GLU per-expert FFNs of width 6144). Headline numbers:

- **Total parameters: 5,711,536,128 (≈5.7B)** — parameter ratio **5.98×** over dense.
- **Activated parameters per token: 1,634,672,640 (≈1.63B)** — 1.71× dense, 3.5× less than the total.
- **Training FLOPs/token: 10,110,025,728 (≈10.1 GFLOPs)** — 1.68× the dense baseline.

The expert count is the constraint-driven decision: no published 1B-dense MoE recipe assumes a 25 Gbps inter-node cap with a fixed EP degree of 4, so this design deliberately shrinks expert count (8, vs the 16–64 typical) to keep the *total* parameter count — and therefore the per-step gradient all-reduce volume crossing the slow interconnect — inside the bandwidth budget. Expert count and top-k are stated in the Parameters table; the bandwidth arithmetic is in Training implications.

## Parameters table

All figures follow the skill's formulas (`ffn_dim = 4·1536 = 6144`; per-expert GLU FFN `= 3·1536·6144 = 28,311,552`; per-layer attention `4·1536² = 9,437,184`; layernorm+router term `2·1536 = 3,072`; embedding `32000·1536 = 49,152,000`). Router weights (8·1536 = 12,288 per layer) are negligible and omitted, per the calculator convention.

| Component (24 layers) | Dense | MoE | Activated |
| --- | --- | --- | --- |
| Attention | 226,492,416 | 226,492,416 | 226,492,416 |
| Expert FFNs | 679,477,248 | **5,435,817,984** (24×8×28,311,552) | **1,358,954,496** (24×2×28,311,552) |
| LayerNorms + router | 73,728 | 73,728 | 73,728 |
| Embedding | 49,152,000 | 49,152,000 | 49,152,000 |
| **Total** | **955,195,392** | **5,711,536,128** | **1,634,672,640** |

- **num_experts = 8**, **top_k = 2** (both explicitly stated as required).
- param_ratio = 5,711,536,128 / 955,195,392 = **5.98×**; activated/dense = **1.71×**.
- FLOPs/token = 6 × 1,634,672,640 + 4·24·1536·2048 = 9,808,035,840 + 301,989,888 = **10,110,025,728**.

## Routing

- **Strategy:** Top-2 (not Top-1). Top-2's extra dispatch doubles the intra-pod all-to-all, which is free on the pod's fast fabric (805 MB/step on NVLink-class links ≈ 8 ms) — so there is no latency reason to drop to Top-1 and halve effective expert capacity. With only 8 experts, each expert is precious; Top-1 forfeits knowledge blending at zero bandwidth benefit.
- **Capacity factor: 1.0.** Expert load is actively balanced (below), so the capacity factor stays at the efficient setting — capacity factor is *not* the pressure valve here; bandwidth is.
- **Auxiliary load-balancing loss: 0.01** (upper end of the 0.001–0.01 band). With a small expert set, collapse of even one or two experts is disproportionately damaging; the stronger aux loss is cheap insurance. Router-logit jitter is recommended at train time for the same reason.

## Training implications

**Cluster and parallelism.** 16 GPUs = 4 pods × 4 GPUs. **Expert-parallel degree is fixed at 4**: DP=4 pods, EP=4, TP=1, PP=1, with DP×TP×PP×EP = 16 ✓ and EP divides num_experts (8/4 = 2 experts per GPU) ✓. Each pod holds a full model replica; experts never leave their pod.

**Token-communication volume per step (required figure).** Geometry: micro-batch 8×2048 = 16,384 tokens/GPU, grad_accum = 32, DP = 4 pods → **2,097,152 tokens per optimizer step**. All-to-all volume per step, using the skill's definition (`top_k × tokens/step × dtype_bytes × EP_degree`, dispatch+receive, per MoE layer, summed over layers):

```
all-to-all/step = 24 layers × 2 × 2,097,152 × 2 bytes (bf16) × 4 (EP) = 805,306,368 B ≈ 805 MB/step
```

**This volume never crosses the 25 Gbps interconnect.** Because EP=4 is pinned to a 4-GPU pod, token dispatch and reduction are entirely intra-pod on the fast fabric (~8 ms/step). The 25 Gbps link is reserved for exactly one thing: **DP gradient all-reduce** between the 4 pods:

```
model bf16 = 5,711,536,128 × 2 = 11,423,072,256 B ≈ 11.4 GB
ring all-reduce over P=4 pods, per-pod link volume = 2·(P−1)/P × size = 1.5 × 11.4 GB ≈ 17.1 GB/step
link time at 25 Gbps (3.125 GB/s) ≈ 5.5 s/step  (8.6 GB / 2.7 s with fp8 gradient exchange)
```

**How this stays within budget.** (1) The dominant MoE traffic — token dispatch — is architecturally zero on the slow link, since EP=4 collapses it inside the pod. Even if a scheduler ever split a pod, one layer's dispatch is only 805 MB/24 ≈ 34 MB ≈ 11 ms on the slow link, so there is large robustness margin. (2) The 25 Gbps budget is consumed only by gradient sync, which is made to fit three ways: the **lean 8-expert design caps total params at 5.7B** (a 16-expert design would add ≈10.9 GB/step ≈ +3.5 s on the link — budget-breaking); gradient exchange uses **fp8 with fp32 local accumulation** (17.1 → 8.6 GB/step, ~2.7 s); and the **large global batch (2,097,152 tokens) stretches step compute to ~9.5 s** at ~35% MFU, so an async/overlapped all-reduce hides the 2.7 s sync at ~29% link duty — within the cap, not saturating it. bf16 exchange (5.5 s, 58% duty) is the documented fallback that still fits.

**Memory (per GPU, H100 80 GB):** params 2.86 GB (5.7B/EP=4), grads 2.86 GB, AdamW-mixed optimizer 17.1 GB, activations ≈6.0 GB with gradient checkpointing (24 × 16,384 × 1536 × 20 / tp, halved). Total ≈33 GB → **~59% headroom**, no OOM risk; expert-redundancy cost (each of 4 pods stores a full 5.7B copy → 22.8B stored cluster-wide) is noted as a checkpoint/storage cost.

## Risks

- **Inter-node gradient sync is the dominant risk.** On 25 Gbps the link time is the same order as step compute; a regression to bf16 exchange, a smaller batch, or more pods without ZeRO-style sharding of the dense components pushes link duty toward 100%. Mitigations: fp8 exchange, grad_accum ≥ 32, async overlap, and (fallback) delayed parameter averaging (sync every k steps). An inter-node sync-time check should gate every batch-schedule change.
- **Low expert count (8) caps capacity.** Fewer experts mean less expert specialization and a higher quality ceiling per unit of added parameter memory, but the count is forced by the bandwidth budget. Mitigations: top-2 blending, aux loss 0.01, jitter — never top-1, never aux < 0.005.
- **Router collapse is more damaging at 8 experts.** Losing one expert costs 12.5% of expert capacity. Mitigations: aux loss 0.01, capacity factor ≥ 1.0 so under-loaded experts keep gradients, conservative LR/warmup.
- **Over-parameterization without compute gain.** Total (5.7B) is 3.5× activated (1.63B); optimizer/checkpoint memory and the storage of 4 redundant replicas scale with *total*, while FLOPs scale with *activated*. If storage or per-GPU memory becomes binding, drop ffn_mult to 2 before touching expert count, since expert count is the bandwidth lever.
- **EP=4 correctness assumption.** The whole bandwidth argument rests on experts staying inside a pod. The cluster scheduler must never place EP ranks across pods; the design adds an intra-pod placement assertion to the launch plan.