# MoE Architecture for a 25 Gbps Interconnect (EP=4, ~1B Dense-Equivalent)

## 1. Overview

Design target: an expert-parallel MoE model of **≈0.96B total parameters (~1B dense-equivalent)** trainable on a cluster whose inter-node interconnect is capped at **25 Gbps (3.125 GB/s per link), no InfiniBand**, with **expert-parallel degree fixed at EP = 4** because GPUs are grouped into fixed pods of 4.

Key decisions that make the design fit the constraint:

- **EP = 4 (mandated).** Each MoE layer shards its experts across the 4 GPUs of one pod. **No tensor parallelism** — TP would push per-layer activations over the same 25 Gbps links; attention stays fully local per GPU.
- **num_experts = 8, top_k = 2.** With EP=4 each GPU hosts exactly E/EP = 2 experts. Top-2 activates 2 of 8 experts (25% of expert params) per token — Mixtral-style routing quality with bounded dispatch volume.
- **MoE in only 12 of 24 layers** (even-indexed); the other 12 use dense FFN. Halves the expert all-to-all volume vs. an all-MoE stack.
- **FP8 (1-byte) dispatch/combine payloads**, cutting expert-token byte volume 4× vs. FP32 (as in DeepSeek-V3-style FP8 communication).
- Expected remote fraction per dispatched token is (EP−1)/EP = 3/4.

**Worst-case assumption:** the 4 GPUs of a pod are reachable only over 25 Gbps links, so every remote expert dispatch crosses the constrained fabric. If pods are single nodes with a faster local bus, expert traffic leaves the 25 Gbps links entirely and the budget has even more headroom.

## 2. Parameters

| Hyperparameter | Value |
|---|---|
| d_model | 1024 |
| num_layers | 24 (12 MoE + 12 dense FFN) |
| d_ff (dense FFN) | 4096 |
| num_heads / head_dim | 16 / 64 |
| **num_experts** | **8** |
| **top_k** | **2** |
| d_ff_expert (SwiGLU) | 2304 |
| vocab_size (tied embedding) | 32,768 |
| **Expert-parallel degree** | **4** (fixed by pod topology) |
| Experts per GPU | 2 |

| Component | Params |
|---|---|
| Embeddings (tied in/out) | 33.6M |
| Attention (24 layers) | 100.7M |
| Dense FFN (12 layers) | 151.0M |
| MoE experts (12 layers × 8 experts) | 679.5M |
| Routers + norms | 0.15M |
| **Total** | **≈ 965M (≈0.96B, ~1B dense-equivalent)** |
| Params per GPU (EP=4) | ≈ 455M (≈0.9 GB BF16) |
| Active params per token (top-2/8) | ≈ 455M (~47% of total) |

## 3. Routing choice + justification

**Choice: top-2, load-balanced, with capacity factor 1.25 and no token dropping.**

Justification against the low-bandwidth constraint:

- **top-1 vs top-2:** top-1 halves dispatch volume but is notoriously unstable and gives worse quality-per-param; with only 8 experts there is little routing diversity to absorb top-1 variance. top-2 matches the Mixtral precedent and is still only 2 dispatch copies per token. The volume math below shows top-2 **fits comfortably**, so there is no need to sacrifice quality.
- **Load balancing:** a soft auxiliary balance loss (weight α ≈ 0.01, DeepSeek-style) keeps per-expert load near uniform, which directly bounds worst-case all-to-all traffic. Expected tokens/expert/layer = T·top_k/E = 8192·2/8 = 2048; capacity 2560 (×1.25) absorbs jitter. No token dropping (unlike Switch) avoids silent quality loss and bounding-box surprises.
- **Remote-hop expectation:** experts are uniform across 4 GPUs; a chosen expert is local with p=1/4, so the expected remote fraction is top_k·(EP−1)/EP = 1.5 remote copies per token (3/4 of dispatch bytes).

## 4. Training implications

Batch: seq_len 2048, micro-batch 4/GPU → **T = 8192 tokens/GPU/step**, global 32,768 tokens/step on the 4-GPU pod. Model+optimizer state (~6 GB/GPU with Adam + master weights) fits on commodity GPUs.

**Token-communication volume per step (expert dispatch + combine, per GPU):**

`Vol_step = 2 × M × T × top_k × d × b × (EP−1)/EP`

with M=12, T=8192, top_k=2, d=1024, EP=4, b = bytes/float:

| b | Per-GPU/step | Time @ 25 Gbps (3.125 GB/s) |
|---|---|---|
| FP32 (4 B) | 1.21 GB | 0.39 s |
| BF16 (2 B) | 0.60 GB | 0.19 s |
| **FP8 (1 B) — chosen** | **0.30 GB** | **0.097 s** |

**Staying within the 25 Gbps budget — full per-GPU step accounting:**

| Traffic | Bytes/GPU/step | Time @ 25 Gbps |
|---|---|---|
| Expert dispatch+combine (FP8, top-2) | 0.30 GB | 0.10 s |
| Gradient allreduce, replicated params* (BF16 grads) | 0.86 GB | 0.27 s |
| **Total inter-node** | **1.16 GB** | **0.37 s** |

*Replicated (non-expert) params = attention 100.7M + dense FFN 151.0M + embeddings 33.6M ≈ 285M; ring-allreduce traffic = 2×285M×2B×3/4. Expert gradients need **no** allreduce — each expert is owned by one GPU (this is the EP payoff: 679M expert params never cross the fabric).

At a target step time of ~3 s the budget is 9.4 GB/GPU/step; we use **1.16 GB ≈ 12%**, and ~15% even at a 2 s step. Expert all-to-all also overlaps with attention/dense-layer compute across microbatches, so the realized wall-clock hit is smaller still.

Scaling to multiple pods adds only DP gradient sync (same 25 Gbps links); keep DP ≤ 4–8 per GB-scale model, or enable gradient compression beyond that.

## 5. Risks

- **All-to-all bursts under skewed routing:** capacity factor + aux loss mitigate, but a severe skew turns dispatch into a straggler. Mitigate with per-expert load monitoring and dynamic capacity raise.
- **FP8 dispatch precision:** minor quality loss possible; BF16 fallback (2× traffic, still within budget) is a one-line knob.
- **Limited routing diversity (E=8):** risk of expert collapse/degenerate specialization; monitor expert utilization, keep the α≈0.01 balance loss.
- **EP=4 is a hard ceiling:** experts are capped at multiples of 4; growing beyond 1B forces deeper/wider per-expert FFNs rather than more experts, since the pod count per EP group is fixed.
- **No TP safety net:** attention cannot be sharded across the 25 Gbps links; if d_model must grow, per-GPU attention memory grows with it.
- **Gradient sync at scale:** beyond ~4 pods, DP allreduce over 25 Gbps grows linearly; mitigate with fewer pods per model, gradient compression, or local-SGD-style reduced sync frequency.

**Bottom line:** EP = 4 (mandated), num_experts = 8, top_k = 2, 12/24 MoE layers, FP8 dispatch → **0.30 GB/GPU/step of expert-token traffic (0.097 s), plus 0.86 GB gradient sync (0.27 s), total ≈1.16 GB/step ≈ 12% of the 25 Gbps per-link budget.**