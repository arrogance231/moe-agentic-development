Here is the architecture document.

---

# MoE Architecture — 1B Dense-Equivalent on a 25 Gbps (No-InfiniBand) Cluster

## 1. Overview

**Hard constraints**
- Inter-node fabric: **25 Gbps Ethernet, no InfiniBand** → ≈ 3.125 GB/s sustained per direction (full duplex).
- **Expert-parallel (EP) degree fixed at 4**: GPUs are grouped into fixed physical pods of 4; experts may not span pods.
- Dense-equivalent target: **≈ 1B parameters**.

**Core design decision — co-locate the EP group inside one pod.**
Every MoE layer's 64 experts are spread over the 4 GPUs of a single pod (EP=4, 16 experts/GPU). Consequently **all expert dispatch/combine token traffic rides intra-pod NVLink and zero bytes of routed-token traffic ever cross the 25 Gbps fabric.** This turns the "low-bandwidth interconnect" problem into a non-problem for the dominant MoE communication.

**Cross-pod parallelism — data-parallel local-SGD, not per-step gradient all-reduce.**
A per-step full-gradient all-reduce of a 1B-scale model is ≈ 3.4 GB/step ≈ **1.1 s/step on 25 Gbps — impossible**. Instead, each pod trains an independent full-model replica and averages weights over the fabric only every T=128 steps (a single coalesced transfer, asynchronously overlapped with training). Scaling comes from adding pods (more DP replicas), never from raising EP.

**Why no published recipe covers this:** existing MoE recipes assume high-bandwidth inter-node fabric for EP all-to-all (GShard, Megatron-EP, DeepSeek), or use top_k=1 (Switch) specifically to shrink cross-node traffic; classic local-SGD is never combined with a fixed small EP=4 pod topology. This design uses the pod topology to eliminate the fabric token tax (so top_k=2 stays affordable) and solves the fabric problem with delayed weight sync.

| Headline quantity | Value |
|---|---|
| Dense-equivalent size | 1,073,741,824 (≈ 1.07 B ≈ target 1B) |
| Total MoE parameters | **1,680,867,328 (≈ 1.68 B)** |
| Active parameters / token | ≈ 443.5 M (≈ 0.44 B), ~3.8× sparsity, ~2.7× fewer FLOPs than dense |
| Layers / hidden dim | 24 / 2,048 (16 heads, head-dim 128) |
| **num_experts** | **64** (per MoE layer) |
| **top_k** | **2** |
| **Expert-parallel degree** | **4 (one pod)** — constraint satisfied exactly |
| Token-comm / step (intra-pod) | 2.4 GB send + 2.4 GB recv per GPU |
| Token-comm / step (fabric) | **0** |
| Sustained fabric utilization | ≤ ~17% (with headroom) |

## 2. Parameters

All 24 FFNs are MoE (dense FFN replaced by 64 experts + router). Expert FFN is narrow: gate/up/down with `d_e = 128` (d=2048 → 128 → 2048). Tied embedding. Norms negligible (not counted).

| Component | # units | params / unit | Total params |
|---|---|---|---|
| Embedding (tied) | 1 | 32,768 × 2,048 | 67,108,864 |
| Attention (Q, K, V, O) | 24 layers | 4 × 2,048 × 2,048 | 402,653,184 |
| MoE router (2048→64) | 24 layers | 2,048 × 64 | 3,145,728 |
| Experts (gate+up+down) | 24 layers × 64 experts | 3 × 2,048 × 128 | 1,207,959,552 |
| **Total (MoE)** | — | — | **1,680,867,328 (≈ 1.68 B)** |
| Active per token | — | attn + 2×24 experts + routers | ≈ 443,547,648 (≈ 0.44 B) |
| Dense-equivalent reference | 24L, d=2048, d_ff=6144, tied emb | — | 1,073,741,824 (≈ 1.07 B) |

**num_experts = 64, top_k = 2, EP degree = 4** (per MoE layer, 16 experts resident per GPU). Capacity factor 1.25.

## 3. Communication & 25 Gbps budget accounting

Step definition: one optimizer step = 16,384 tokens per pod (64 micro-batches × 256 tokens; global batch = P × 16,384 over P pods). bf16 token = 2,048 floats × 2 B = 4,096 B.

**Token-communication volume per step** (all-to-all, dispatch + combine):

- Per MoE layer, per GPU, tokens leaving to the 3 other GPUs of the pod: `N × top_k × (EP−1)/EP = 16,384 × 2 × 0.75 = 24,576 tokens` → `24,576 × 4,096 B ≈ 100 MB` send (100 MB recv symmetric).
- × 24 MoE layers → **2.4 GB sent + 2.4 GB received per GPU per step; 9.6 GB + 9.6 GB per pod**. All on intra-pod NVLink (≥100 GB/s/GPU) ≈ 48 ms/step.

**Fabric (inter-node) traffic — the only thing the 25 Gbps budget sees:**

- Weight averaging every T=128 steps: 1.68 B params × 2 B = **3.36 GB per all-reduce**, done as one coalesced two-level transfer (intra-pod NVLink reduce, then one large fabric message). Amortized = **26.3 MB/step**.
- Global scalars (LR, grad-norm for clipping): < 1 KB/step.
- **Sustained ≈ 26 MB/step → ≈ 4.2 Gbps at a 50 ms step (~17% of the 25 Gbps budget, ~83% headroom); burst of 25 Gbps for ~1.1 s every ~6.4 s, fully overlapped with local compute.**

| Traffic type | Volume / step / pod | Path | Fits 25 Gbps? |
|---|---|---|---|
| Expert token dispatch + combine | 9.6 GB send + 9.6 GB recv | intra-pod NVLink | off-fabric (never on the link) |
| Weight average (T=128) | 26.3 MB amortized | fabric | ✓ ≈ 4.2 Gbps sustained |
| Grad-norm / LR scalars | < 1 KB | fabric | ✓ |
| Per-step gradient all-reduce | (3.4 GB) | fabric | ✗ rejected (~1.1 s/step) |
| Pipeline activation pass | (n/a — no PP) | fabric | n/a |

**How the budget is kept:** the dominant MoE traffic is the token all-to-all. Forcing it onto 25 Gbps would cost `2.4 GB ÷ 3.125 GB/s ≈ 770 ms/step` — ~15× the entire budget — so EP=4 being pinned to a pod is treated as a *requirement* that removes it from the budget entirely. Everything that does cross the fabric is deliberately slow-frequency (weight averaging, not per-step gradients), large-message (bandwidth-bound, insensitive to Ethernet latency), and overlapped with compute.

## 4. Routing choice and justification

**Choice: softmax top-2 token-choice (GShard/Megatron style).** Router linear 2048→64 → softmax over 64 experts → top-2. Training-time jitter noise (σ≈0.05), auxiliary load-balancing loss (coefficient 0.01), capacity factor 1.25 with token dropping.

- **Why not top_k=1 (Switch):** top-1 exists to halve cross-node all-to-all traffic. Here that traffic is intra-pod and free, so top-1 buys nothing (saves ~24 ms/step intra-pod) while forfeiting top-2's robustness to load imbalance and its redundancy against a single saturated expert.
- **Why not expert-choice routing:** it needs global per-expert capacity planning and cross-GPU token redistribution — coordination whose only payoff (better balance under heavy cross-node traffic) is nullified by the intra-pod design.
- **Why the load-balancing loss matters:** with 64 experts over 4 GPUs (16/GPU), imbalance would create intra-pod stragglers and capacity-factor drops. The aux loss keeps utilization smooth, keeps drops ≈ 0, and — critically — prevents any *cross-pod* token push that would put routed-token bursts on the 25 Gbps fabric. Cross-pod expert placement is forbidden by design, not just topology.

## 5. Training implications

- **Parallelism:** DP across pods (local-SGD), EP=4 intra-pod, TP=1. P pods ⇒ P× throughput; EP does **not** grow with GPU count (per constraint).
- **Optimizer/memory:** AdamW, bf16 fwd/bwd, fp32 master + fp32 m/v, all resident on the GPU that owns the parameters (each pod = full replica). Per-GPU resident ≈ 0.62 B params ≈ **~10 GB** (weights+grads+optimizer) — fits 40 GB and 24 GB GPUs.
- **Data:** each pod reads a disjoint shard; 16,384 tokens/pod/step, sequence packing, 64 × 256-token micro-batches.
- **Sync protocol:** every T=128 steps, two-level all-reduce of weight deltas: intra-pod NVLink reduce (free), then one coalesced 3.36 GB fabric transfer per pod. Runs asynchronously while the next 128 steps train on slightly stale weights; nothing on the critical path is fabric-bound.
- **Overlap:** intra-pod dispatch/combine overlapped with attention GEMMs; fabric sync overlapped with training; compute per step ≈ 11 TFLOP/pod (~10 ms on 4×H100-class at ~30% MFU), so steps are dominated by intra-pod all-to-all (≈50 ms), not the fabric.
- **Checkpointing:** each pod writes its replica to shared storage every few sync windows; recovery restores from the last completed sync (all pods consistent). Checkpoints scheduled *between* sync bursts.
- **Determinism:** identical init seed and dataloader sharding across pods; token-dropping RNG seeded identically so replicas stay bit-compatible before averaging.

## 6. Risks

1. **Local-SGD staleness** — delayed averaging can slow or destabilize convergence. Mitigate: T=128 with LR schedule, eval monitoring; reduce T on drift (cost is amortized fabric traffic).
2. **Sync bursts** — full-bandwidth bursts (~1.1 s) every ~6.4 s. Mitigated by async overlap + two-level reduce; residual risk if the fabric is co-tenanted (checkpoints/logging) — keep sustained ≤ ~20%.
3. **Scaling ceiling** — all-reduce latency grows ~log P; beyond ~16 pods use a hierarchical tree and/or delta compression (bf16→int8 weight deltas, 1-bit Adam) to hold amortized traffic flat.
4. **EP=4 is a hard ceiling** — adding pods cannot reduce per-GPU expert memory. Growth path is more experts/layers (raise E or d_e), not more parallelism.
5. **Load imbalance → dropped tokens → quality loss.** Capacity 1.25 absorbs typical imbalance; worst case requires raising capacity (intra-pod traffic, still free) rather than cross-pod resend.
6. **Narrow experts (d_e=128)** may underfit knowledge-dense tasks vs the dense 1B baseline. Monitor; fallback is d_e=256 (total → ≈3.2 B) or E=128.
7. **Intra-pod bandwidth assumption** — if a "pod" is PCIe 4.0 x16 only (32 GB/s), intra-pod all-to-all grows to ~150 ms/step; still ≥10× the fabric, but verify NVLink on the actual 4-GPU nodes.
8. **Ethernet latency** — high RTT punishes fine-grained/many-message collectives; the single coalesced 3.36 GB transfer per sync is bandwidth-bound and latency-insensitive. Avoid latency-sensitive schedules (e.g., per-micro-batch cross-pod sends).
9. **Fabric co-tenancy** — keep sustained utilization ≤ ~20% so checkpoint/eval bursts never collide with a sync burst.
10. **Convergence determinism under token dropping** — mismatched drop RNG across pods would break replica agreement before averaging; enforce identical seed.

**Constraint compliance checklist:** EP degree = 4 ✓ · num_experts = 64, top_k = 2 ✓ · total params ≈ 1.68 B (dense-equivalent ≈ 1.07 B ≈ 1B) ✓ · token-communication volume/step = 2.