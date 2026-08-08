## MoE Architecture Document — 1B Dense-Equivalent on a 25 Gbps Interconnect

All figures below are hand-applications of the exact formulas in `tools/moe_calculator.py` (ffn_dim = ffn_mult·d_model; attention = 4·d_model²; per-expert GLU FFN = 3·d_model·ffn_dim; embedding = vocab·d_model; moe_params = layers·(4·d_model² + 2·d_model + num_experts·3·d_model·ffn_dim) + vocab·d_model; activated = same with top_k in place of num_experts), with arithmetic shown so every digit is independently checkable.

### Overview

The dense-equivalent baseline is a **1.086B-parameter** dense LM (20 layers, d_model 1792, ffn_dim 7168, vocab 32,768). The MoE replaces the per-layer dense FFN with **8 experts and top-1 routing**. Total parameters **6,481,319,936 (6.48B)**; activated parameters **1,086,396,416 (1.086B)** — exactly the dense-equivalent target; parameter ratio **5.97×**. Because top-1 activates one expert per token, training FLOPs are **identical to the dense baseline** (6.81 GFLOP/token): the design converts the same compute budget into ~6× parameter capacity. **num_experts = 8, top_k = 1, expert-parallel degree = 4.** The EP group of 4 exactly equals the fixed pod size, so all expert token dispatch runs on the intra-pod interconnect and the 25 Gbps inter-node link is never used for all-to-all during a training step.

### Model architecture

| Hyperparameter | Value |
| --- | --- |
| num_layers | 20 |
| d_model | 1792 |
| ffn_mult (per-expert) | 4 → ffn_dim 7168 |
| **num_experts** | **8** |
| **top_k** | **1** |
| vocab | 32,768 |
| seq_len | 2048 |
| capacity factor | 1.25 |
| aux loss (load-balance) | 0.01 |
| precision | BF16 |

### Parameters table

| Component | Dense | MoE (8 experts) | Activated (top-1) |
| --- | --- | --- | --- |
| Attention (20 layers, 4·1792² each) | 256,901,120 | 256,901,120 | 256,901,120 |
| Expert FFNs (20 × 3·1792·7168 × {1, 8, 1}) | 770,703,360 | 6,165,626,880 | 770,703,360 |
| Layer norms (20 × 2·1792) | 71,680 | 71,680 | 71,680 |
| Embedding (32,768 × 1792) | 58,720,256 | 58,720,256 | 58,720,256 |
| Router (8 × 1792 × 20) | — | 286,720 (negligible, omitted) | 286,720 (negligible) |
| **Total** | **1,086,396,416** | **6,481,319,936** | **1,086,396,416** |

- param_ratio = 6,481,319,936 / 1,086,396,416 = **5.97×**
- Per-expert FFN = 3·1792·7168 = 38,535,168; attention per layer = 12,845,056; embedding = 58,720,256
- With top-2 the activated total would be 1,857,099,776 (1.86B) — outside the "1B dense-equivalent" spec.

### Routing

- **Strategy: Top-1.** Justification: (1) top-1 activated params = 1.086B, matching the 1B dense-equivalent target exactly (top-2 would activate 1.86B); (2) top-1 halves token-communication volume versus top-2 (167.8 MB/step vs 335.5 MB/step, §Training implications), directly serving the 25 Gbps budget; (3) top-1 keeps per-token FLOPs equal to the dense baseline.
- **top_k = 1**, capacity factor **1.25** (absorbs token-routing imbalance — 8,192 tokens/expert/GPU/micro-batch makes drops unlikely), aux loss **0.01** load-balancing (+ router z-loss 0.001) to prevent collapse.

### Training implications

**Parallelism.** EP = 4 (fixed by pod topology), TP = 1, PP = 1, DP = 1 per training job; product 4 = pod GPU count; EP divides num_experts (2 experts/rank). Batch geometry: micro_batch 8, grad_accum 64 → global batch 512 seqs = **1,048,576 tokens/step**; tokens per expert per GPU per micro-batch = 8·2048·1/2 = 8,192 ≫ 8–64 floor.

**Compute.** 6.81 GFLOP/token = dense-baseline FLOPs. On one pod (4×H100, ~40% MFU ≈ 1.58 PFLOP/s): step ≈ 4.5 s, ≈ 232k tokens/s. Memory/rank (80 GB): bf16 weights 3.71 GB + fp32 grads 7.43 GB + AdamW 22.29 GB + activations ~5.9 GB (gradient checkpointing) ≈ 45 GB → 44% headroom (over budget on 40 GB cards; drop micro_batch to 4 or enable ZeRO).

**Constraint: EP = 4 and token-communication volume within 25 Gbps.**
- Token-communication (all-to-all) volume per step: `top_k × tokens_per_step × dtype_bytes × experts_involved` per MoE layer = 1 × 1,048,576 × 2 (bf16) × 4 (EP) = **8,388,608 B (8.39 MB) per layer**, × 20 layers = **167,772,160 B ≈ 167.8 MB per step**.
- Budget check: 25 Gbps = 3.125 GB/s. 167.8 MB / 3.125 GB/s = **54 ms ≈ 1.2% of the 4.5 s step** even in the worst case where the full volume crossed the link.
- How it stays within budget: the EP=4 group is placed entirely **inside one 4-GPU pod**, so dispatch/receive all-to-all runs on the intra-pod interconnect (NVLink-class, ≥10× faster than 25 Gbps) and **consumes none of the 25 Gbps budget**. The inter-node link carries only checkpointing and telemetry.
- Scaling caveat: cross-pod DP (e.g., 8 pods, DP=8) is **not** viable on this interconnect — a gradient ring all-reduce of 6.48B params costs 2·(7/8)·6,481,319,936·4 B ≈ 45.4 GB/step → 14.5 s/step at 25 Gbps, 3.2× the entire single-pod step. The pod is the unit of scaling: one training job per pod.

### Risks

1. **Top-1 quality loss vs top-2** (no expert blending). Mitigation: 8 diverse experts with aux loss keep top-1 capacity utilized; fallback documented — a top-2 variant (activated 1.86B, 335.5 MB/step all-to-all ≈ 2.4% of step time on-link) still fits the interconnect; the spec, not the link, is what forces top-1.
2. **Router collapse** under top-1. Mitigation: aux loss 0.01, z-loss, capacity factor 1.25, logit jitter 0.01.
3. **Over-parameterization on 4 GPUs.** 6.48B total/1.86B per rank fits 80 GB with 44% headroom; 16 experts would raise per-rank memory to ~80 GB and OOM — 8 is the memory-bounded maximum for a 4-GPU pod.
4. **Expert imbalance → padding waste.** Mitigation: load-balancing aux loss and jitter; dynamic capacity if skew persists.
5. **Cross-pod scaling trap.** Naive DP across pods blows the 25 Gbps budget (14.5 s/step gradient all-reduce); the design mandates pod-local parallelism and one job per pod.