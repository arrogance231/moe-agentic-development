Here is the architecture document. (Design assumptions stated up front so the constraint accounting is checkable.)

---

# MoE Architecture for a 25 Gbps, Fixed EP=4 Cluster

**Constraint statement.** Inter-node interconnect = 25 Gbps (3.125 GB/s), no InfiniBand. GPUs are grouped into fixed pods of 4, so **expert-parallel degree = 4 (EP = 4) regardless of GPU count**; adding pods only adds data-parallel replicas. Target dense-equivalent size = 1B. There is no published recipe for this combination, so the design below stitches together standard parts (fine-grained experts, top-2 gating, fp8 dispatch) tuned specifically to the wire budget.

> **Fabric assumption (worst case, designed-for).** All GPU-to-GPU traffic — including expert dispatch/combine — crosses the 25 Gbps fabric; a "pod" is only a routing/topology grouping, not a faster internal fabric. If intra-pod fabric were faster, the budget below only gets easier. This makes the design safe under the strictest reading of the constraint.

## Overview

A 24-layer transformer with every-layer MoE. Attention, routers, and embeddings are dense (shared across DP replicas); each MoE layer holds 8 experts, 2 of which are activated per token. EP=4 places exactly 2 experts per rank, so all-to-all fan-out is small and deterministic. To keep per-step token traffic inside the 25 Gbps budget, the dispatch/combine tensors are sent in fp8 (1 byte/dim) while compute stays bf16, and the micro-batch is sized so a full step's all-to-all completes in ~1 s.

## Parameters

| Item | Value |
|---|---|
| Hidden dim `d` | 2048 |
| Layers `L` | 24 (all MoE) |
| **num_experts `E`** | **8** |
| **top_k** | **2** |
| Expert FFN inner dim `d_exp` | 2730 |
| Per-expert params (gate/up/down) | 3 · 2048 · 2730 = 16.77M |
| Expert params per layer (×8) | 134.2M |
| Attention params per layer | 16.8M |
| Layer total (incl. router ≈0.02M) | ≈ 151.0M |
| Embeddings (vocab 32,768 × 2048) | 67.1M |
| **Total parameters** | **≈ 3.69B** (24 · 151.0M + 67.1M) |
| Activated params / token | 24 · (16.8M + 2·16.77M) + 67.1M ≈ **1.28B** |
| Dense-equivalent size | **≈ 1.2B** (activated FLOPs match a 1B dense model) |

**Why `E=8`.** EP=4 gives 2 experts/rank. `E=8` keeps total params ≈3.7B (≈3.7× dense) and on-rank expert memory low; `E=16` would double the table to ≈7.4B with zero communication benefit, since all-to-all volume depends on `top_k·d`, not `E`. **Why `top_k=2`.** 2 activated experts (33.5M/layer) reproduce exactly the FFN FLOPs of a 1B dense model, and 2 dispatch copies is the minimum that preserves the dense-equivalent quality/FLOP contract (top-1 cuts comm in half but measurably degrades MoE quality).

## Routing choice

**Top-2 learned gating (softmax router) + auxiliary load-balancing loss, capacity factor 1.0.**

Justification:
- With only 2 experts/rank, a hot expert would strand half of its rank's expert compute and serialize traffic on the wire; the aux loss keeps per-expert utilization uniform, which is the single most important thing on a 25 Gbps fabric.
- `top_k=2` spreads each token to 2 of the 3 *other* ranks on average, giving a balanced all-to-all traffic matrix across the EP group.
- Expert Choice routing was evaluated as an alternative (it removes the aux loss and the imbalance risk) but can drop tokens when capacity is pinned at 8 experts; it is retained as the fallback under Risks.

## Training implications

**EP=4, DP=4, PP=1** on a 4-pod/16-GPU cluster (more pods → more DP replicas, EP stays 4). Micro-batch `B = 16,384` tokens/step (8 × 2048).

**Token-communication volume per step** (expert all-to-all, dispatch + combine, per MoE layer):

```
per layer per token = top_k · d · bytes · 2 (round trip)  = 2 · 2048 · 1 · 2 = 8 KB
per step (24 layers) = 24 · 8 KB · B = 192 KB · B
for B = 16,384  →  V_all2all ≈ 3.15 GB/step
```
(bf16 would be 6.29 GB/step; fp8 dispatch is why we stay at 3.15 GB.)

**Budget check (25 Gbps = 3.125 GB/s).**
- All-to-all: 3.15 GB → ~1.0 s at peak link rate 25 Gbps (instantaneous rate reaches the cap, duty ≈25% of a ~4 s step).
- DP gradient all-reduce: experts are EP-sharded (no cross-replica sync); only shared params (≈470M attn+routers+embeddings) sync → 1.41 GB → ~0.45 s.
- **Total cross-fabric traffic ≈ 4.6 GB/step → ≈1.45 s of a ~4 s step (~35% duty), peak instantaneous 25 Gbps, never exceeding the cap.** Sizing `B` so all-to-all completes in ~1 s is what keeps the design inside budget.

**Throughput consequence (explicit).** The wire caps aggregate throughput at ≈16.4K tokens/s (≈1.4B tokens/day); a 100B-token run ≈ 70 days. This is the price of the constraint and is why no published recipe exists — compute is deliberately under-utilized and we accept a comm-bound regime, using gradient accumulation (64 micro-steps → 1.05M statistical batch) and overlapping dispatch of layer `i+1` with compute of layer `i`.

## Risks

- **Throughput wall (~16K tok/s).** Mitigations if unacceptable: top-1 routing, fp8→ int8 further, or MoE in only 8/24 layers (cuts volume 3×).
- **Expert imbalance strands an EP rank** (2 experts/rank only) → worst-case +1 full all-to-all round trip. Mitigate with aux-loss tuning or switch to Expert Choice routing.
- **Link oversubscription** if several replicas' all-to-alls share the same physical 25 Gbps links — require distinct pod placement for DP replicas.
- **fp8 dispatch quantization** can shift routing quality → monitor perplexity/evals; keep bf16 compute, per-layer scale factors.
- **Latency-sensitive all-to-all** on non-RDMA 25 Gbps: use chunked/pipelined dispatch and batched messages, never per-token round trips.
- **Assumption risk:** if "pod" actually means a faster internal fabric, the all-to-all no longer crosses 25 Gbps and the real bottleneck becomes the shared-param gradient sync (~0.45 s/step) — still fits.