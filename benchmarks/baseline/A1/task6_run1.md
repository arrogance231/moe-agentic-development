# MoE-1B-Pod — MoE Architecture for a 25 Gbps, Fixed 4-GPU-Pod Cluster

## Overview

**Goal.** A ~1B dense-equivalent (sparsely activated) MoE LM that trains correctly and efficiently on a cluster whose only inter-node fabric is **25 Gbps Ethernet (no InfiniBand, no RDMA)** and whose GPUs are physically locked into **fixed pods of 4**. Total GPU count is otherwise free; a representative 32-GPU (8-pod) configuration is used for all per-step arithmetic below.

**Constraining premise.** No published MoE recipe (Mixtral, DeepSeek, Switch, GShard…) assumes the combination of a ~25 Gbps interconnect *and* a forced expert-parallel degree of 4. That combination is handled here as a first-class design input rather than an accident.

**The one decision everything hangs on.** Expert-parallel degree is **fixed at 4**, and 4 is exactly the pod size. An expert-parallel group therefore **never spans nodes**: every expert is reachable inside one pod, so the entire token dispatch/combine all-to-all runs on the pod-internal high-bandwidth fabric and **does not consume the 25 Gbps budget at all**. This turns a supposed limitation into the design's main lever. All other choices (small `num_experts`, `top_k=1`, modest total parameters, fp8 + low-bit gradient sync) are consequences of keeping the *residual* cross-pod traffic — dominated by DP gradient all-reduce — inside the same budget.

**Key numbers.** Total parameters **≈ 5.34B**; activated per token **≈ 1.11B**; token-communication volume **12.9 GB per GPU per step** (51.5 GB per 4-GPU pod), of which **0 GB crosses the 25 Gbps fabric**; cross-pod traffic **≈ 2.7 GB/step** after compression, fitting in a ~6 GB per-step budget at 25 Gbps.

---

## Parameters

| Quantity | Value |
|---|---|
| `d_model` | 2048 |
| Layers `L` | 24 |
| Vocab (tied embeddings, RoPE, no learned positions) | 50,000 |
| **`num_experts` `E`** (per layer) | **8** |
| **`top_k`** | **1** |
| Expert FFN | SwiGLU, intermediate 4096 (= 2×`d_model`) |
| Router | per-layer linear `d_model→E`, aux load-balancing loss, replicated on all 4 GPUs of a pod |
| **Expert-parallel degree (fixed)** | **4** (one EP group = one pod; 2 experts per GPU, no intra-expert sharding) |
| Precision | bf16 weights/activations; fp8 gradients; low-bit compressed sync |
| **Total parameters** | **≈ 5.34B** |

Parameter breakdown (all digits):

| Component | Per layer | × Layers | Total |
|---|---|---|---|
| Tied token embedding + LM head | — | — | 102,400,000 (102.4M) |
| Attention (Q,K,V,O) | 4·2048² = 16,777,216 | 24 | 402,653,184 (402.7M) |
| Experts (8 × SwiGLU 2048→4096→2048) | 8 × 25,165,824 = 201,326,592 | 24 | 4,831,838,208 (4.83B) |
| Routers + norms | ~20,400 | 24 | ~0.5M |
| **Total** | | | **≈ 5,336,984,576 (≈ 5.34B)** |

**Activated (dense-equivalent) size per token:** embedding 102.4M + attention 402.7M + 1 expert × 24 layers = 604M + router ~0.4M → **≈ 1.11B ≈ target 1B**. The sparse FFN activates 604M of 4.83B expert weights per token (~1/8 sparsity at the FFN level).

---

## Routing Choice with Justification

**Routing: token-choice, deterministic `top_k=1` (greedy top-1 after a short router-noise warmup), + Switch-style auxiliary load-balancing loss (coeff ≈ 0.01) and router z-loss.**

- **`top_k=1` is the cheapest possible dispatch.** Per-step token-communication volume is proportional to `tokens × top_k × d_model × bytes`. `top_k=2` (Mixtral-style) would double dispatch to **25.8 GB/GPU/step**; with `top_k=1` it is **12.9 GB/GPU/step**. Under a bandwidth-constrained design, this is a free ~2× communication reduction with only a modest quality cost for an 8-expert pool.
- **One-hot routing means a pure-permutation all-to-all.** There is no multi-expert softmax combine, so the combine phase is a plain gather, not a weighted reduction — the simplest, lowest-overhead all-to-all pattern available and the least demanding on the fabric.
- **`E=8`, not 16/64.** For fixed `top_k`, the *dispatched* volume does not depend on `E`, but **total parameters — and therefore cross-pod gradient bytes on the 25 Gbps link — scale linearly with `E`** (experts hold 90% of the model). `E=8` is the smallest pool that still gives meaningful routing diversity while keeping total params at 5.34B so that fp8 + compression brings DP sync inside budget.
- **Load balance is enforced, not assumed.** `top_k=1` is more collapse-prone than `top_k=2`, so an auxiliary load-balancing loss plus an expert capacity factor (1.0 during steady state, with a small capacity-margin during warmup to avoid token drops) prevents router collapse and unbalanced expert compute.

---

## Training Implications

Representative config: **32 GPUs = 8 pods × 4 GPUs**; EP=4 inside every pod, **DP=8 across pods**; no tensor parallelism and no pipeline stages across pods (they would put activation traffic on the 25 Gbps links for no benefit at this size). Global batch = **1,048,576 tokens/step** (seq len 2048; 128 seq/GPU), chosen to amortize the fixed per-step sync cost.

**Token-communication volume per step (stated explicitly).**
- Per GPU local tokens/step: 1,048,576 / 8 / 4 = **32,768**.
- Per GPU per layer: dispatch + combine + backward dispatch + backward combine = 4 × (32,768 × 2048 × 2 B) = **536.9 MB**.
- **Per GPU per step: 12.9 GB**; **per 4-GPU pod (EP group): 51.5 GB**.

**How this stays inside the 25 Gbps budget.** 25 Gbps ≈ 3.125 GB/s. The design does not put this traffic on the interconnect at all: with EP=4 = pod size, the entire 12.9 GB/GPU/step all-to-all is **intra-pod** (NVLink-class fabric, ~0.2–0.3 s) and **0 B/step crosses the 25 Gbps link**. Routers are replicated (0.4M params), so routing incurs zero cross-pod communication. The 25 Gbps fabric is therefore reserved for exactly one thing:

**Cross-pod traffic = DP gradient all-reduce only.** Raw volume: 2 × 5.34B × 2 B = **21.4 GB/GPU/step** (6.8 s at 25 Gbps — too slow). It is brought into budget by two steps: (1) **fp8 gradients** (÷2 → 10.7 GB), and (2) **low-bit compressed sync** (1-bit Adam / PowerSGD r≈4, with error feedback; ÷8 → **≈2.7 GB ≈ 0.85 s**). Done hierarchically: intra-pod reduce-scatter on NVLink, then the compressed cross-pod all-reduce.

**Budget check (per ~2 s step).** Budget at 25 Gbps over a 2 s step ≈ 6.25 GB. Traffic actually crossing the fabric ≈ **2.7 GB** (compressed grads) — within budget with ~2.3× margin, and fully overlappable with compute (compute step ≈ 1.0–1.5 s at ~6.98e15 FLOP/step over 32 GPUs at ~30–45% MFU) and with the intra-pod all-to-all (~0.25 s). Expected throughput ≈ 0.5M tokens/s across 32 GPUs.

**Additional implications.**
- **Gradient compression is mandatory, not optional** — this is the single non-standard training technique the fabric forces.
- Large global batch (1M tokens) + gradient accumulation amortize the fixed sync latency.
- Per-GPU memory: ~10.7 GB weights (bf16) + fp8 Adam states (~16 GB) + fp8 grads (~5.3 GB) + activations (selective recomputation) ≈ 40–50 GB. If HBM < 48 GB, use ZeRO-2/3 **within** the pod (keeps cross-pod volume unchanged).
- Optionally, locally-asynchronous / delayed gradient sync (every k steps) can trade a little quality for more headroom on the fabric.

---

## Risks

1. **Pod locality assumption.** The "0 GB on fabric" result requires each 4-GPU pod to sit on one node with a fast shared fabric. If pod-internal links are also 25 Gbps, the 12.9 GB/GPU/step all-to-all alone costs ≈ 4.1 s/step — a ~4× throughput hit. Mitigation: shrink global batch to 256K tokens/step (→ ≈1 s all-to-all) at the cost of sync amortization, or accept slower steps.
2. **Compression-induced convergence drift.** 1-bit/PowerSGD sync can degrade quality. Mitigate with warmup, error feedback, and eval monitoring; fallback is fp8-only sync (10.7 GB → ~3.4 s, longer but safe steps).
3. **Router collapse with `top_k=1`.** Needs careful auxiliary-loss tuning and a capacity factor; monitor routing entropy. Dropped tokens (capacity overflow) silently hurt quality.
4. **Expert underutilization / low MFU.** With E=8 and 2 experts/GPU, per-expert GEMMs are small; keep the global batch ≥1M tokens and use batched/fused all-to-all to keep expert kernels dense.
5. **Memory ceiling on 4-GPU pods.** 5.34B params × (weights + optimizer + grads) can exceed small-GPU HBM; intra-pod ZeRO sharding is the mitigation but adds intra-pod chatter.
6. **Throughput ceiling is inherent.** ~0.5M tokens/s on 32 GPUs is the practical ceiling for this fabric; scaling up means adding DP pods (per-node sync cost is unchanged) rather than faster per-step convergence.