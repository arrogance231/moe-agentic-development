I'll start by checking the local context and grounding key conventions (EP communication volume, "dense-equivalent" usage) before writing the design doc.
Grounding confirmed: all-to-all EP dispatch is the dominant MoE bottleneck (~34% of step time, DeepSpeed-MoE), and DeepSeek-V3 specifically eliminated cross-node all-to-all via *group-limited gating* (route tokens only to experts in the same node-local group). That is exactly the lever this design pulls. Here is the architecture document.

---

# Architecture: "PodMoE-1B" — a 1B dense-equivalent MoE for a 25 Gbps, pod-of-4 cluster

## 0. Constraint statement and design thesis

| Constraint | Value |
|---|---|
| Inter-node interconnect | 25 Gbps / node (≈ 3.125 GB/s, no InfiniBand) |
| Expert-parallel degree (fixed) | **EP = 4** (cluster groups GPUs into fixed pods of 4) |
| Target dense-equivalent size | **1B parameters** (active per token) |

**Thesis.** Expert-parallel all-to-all (token dispatch/combine) is the #1 MoE communication cost — it accounts for ~34% of step time in DeepSpeed-MoE and grows as ~(N−1)/N of tokens crossing device boundaries. The standard remedy at hyperscale (DeepSeek-V3) is *group-limited gating*: restrict each token's experts to one node-local group so all-to-all never leaves the node. This design applies that principle at the pod scale:

1. **Set EP = 4 and place the EP group inside one pod.** The 4 GPUs that share all 32 experts sit on the same node, so **100% of the token-communication volume is intra-node NVLink traffic and never touches the 25 Gbps link.**
2. **Data-parallel across pods.** Each pod is a full replica of the same EP-4-sharded model. The only data that must cross the 25 Gbps link is cross-pod gradient synchronization, which is compressed ~30× with error feedback.
3. **Dense-equivalent = active params ≈ 1.0B** (attention + top-2 experts + shared expert per token), total ≈ 7.0B — the Mixtral convention (total ≫ active).

---

## 1. Overview

Decoder-only Transformer with sparse MoE FFNs, sized so the **per-token active compute matches a ~1B dense model** (2·10⁹ FLOPs/token class):

- **24 layers**, hidden width **d = 2048**, 16 heads × 128 head-dim (MHA).
- Each FFN replaced by a MoE block: **32 routed experts + 1 always-on shared expert**, each expert a SwiGLU FFN with intermediate width 2048 (= d).
- **top_k = 2** softmax routing (Mixtral-style) with a DeepSeek-style load-balancing auxiliary loss.
- **EP = 4** shards the 32 experts across the 4 GPUs of a pod (8 experts/GPU/layer); attention, router, shared expert, and embeddings are replicated within the pod (d=2048 needs no TP).
- **DP across pods**: every pod is a full replica of the same 4-way-sharded model; no cross-pod token routing exists by construction.
- All training in BF16 with FP32 masters; inter-pod gradients compressed ~30× (top-k + INT8 + error feedback, or 1-bit Adam) before ring all-reduce.

---

## 2. Parameters table

| Parameter | Symbol | Value | Notes |
|---|---|---|---|
| Hidden dimension | `d_model` | 2,048 | |
| Layers | `L` | 24 | |
| Attention heads / head dim | | 16 × 128 | MHA; no TP needed |
| **Num routed experts** | `E` | **32** | per MoE layer |
| **top_k (active experts/token)** | `k` | **2** | |
| Shared (always-on) experts | | 1 | per layer, captures common computation (DeepSeek-V3) |
| Expert FFN intermediate | `d_ff` | 2,048 | = d_model |
| Vocab | `V` | 100,000 | |
| Context length | `T_seq` | 2,048 | |

**Parameter counts (digits):**

| Component | Per layer | ×L | Total |
|---|---|---|---|
| Attention (QKV+O, 4·d²) | 16,777,216 | ×24 | 402,653,184 |
| Router (d·E) | 65,536 | ×24 | 1,572,864 |
| Routed experts (32 × 2·d·d_ff) | 268,435,456 | ×24 | 6,442,450,944 |
| Shared expert (2·d·d_ff) | 8,388,608 | ×24 | 201,326,592 |
| **Total non-embedding** | | | **7,048,003,584 ≈ 7.05B** |
| Embedding (V·d) | 204,800,000 | — | 204,800,000 |
| **Total incl. embeddings** | | | **7,252,803,584 ≈ 7.25B** |
| **Active / dense-equivalent per token** (non-emb) | attn 402.65M + 2×8.39M×24 + shared 201.3M | | **1,008,205,824 ≈ 1.01B** |

Per-expert = 2·2048·2048 = **8,388,608 (8.39M)** params; active FFN per layer = 2 × 8.39M = 16.78M.

> **Dense-equivalent target met:** active (non-embedding) parameters per token ≈ **1.01B ≈ 1B**. Total is 7.0B — MoE "memory-hungry / compute-cheap" tradeoff, consistent with Mixtral 8×7B (47B total / ~13B active) at 1B scale.

---

## 3. Interconnect budget: EP degree and token-communication volume

**Expert-parallel degree used: EP = 4** (fixed by the constraint), with the 4-way EP group collocated *inside one pod*.

**Token-communication volume per step.** Every MoE layer performs one all-to-all dispatch (tokens → experts) and one all-to-all combine (results → home GPU); dispatch and combine move identical volumes.

- Per token per layer: `2 ops × k × d × 2 bytes = 2 × 2 × 2048 × 2 = 16,384 B (16 KiB)`.
- Global step batch: `B = 8,388,608 tokens` (4,096 seq × 2,048) across all pods.
- **Total token-communication volume per step (global) = 8.39e6 × 24 × 16 KiB ≈ 3.30 TB.**
- Per pod (8-pod cluster) = **412 GB/step**; per GPU = 103 GB/step.

**Where this volume travels — the entire design hinges on this:**

| Traffic | Path | Volume/step (per node) | Time |
|---|---|---|---|
| Token dispatch+combine all-to-all | **intra-pod NVLink** (~200 GB/s) | 412 GB | ≈ 2.1 s |
| Same volume on the 25 Gbps link, *hypothetically* | inter-node | 412 GB | ≈ **132 s → impossible** |
| Gradient all-reduce, uncompressed fp16 (DP=8 ring, egress ≈ 2·14.1 GB·⅞) | **inter-node 25 Gbps** | 24.7 GB | ≈ 7.9 s |
| Gradient all-reduce, **~30× compressed** (top-k + INT8 + error feedback) | inter-node 25 Gbps | **≈ 0.8 GB** | **≈ 0.26 s** |

**How the 25 Gbps budget is satisfied:**

1. **Token communication is taken off the 25 Gbps link entirely.** Because EP = 4 and the pod = 4 GPUs on one node, the 3.30 TB/step of token traffic is carried by NVLink-class fabric (≥ ~200 GB/s, ~50–100× the inter-node link). If the same traffic crossed the 25 Gbps link it would cost ~132 s/step — that is the failure mode this design eliminates. This is exactly the mechanism of DeepSeek-V3's group-limited gating, applied at pod granularity; here the "expert group" is the pod, so group-limited routing is automatic.

2. **The only inter-node traffic is DP gradient sync**, bounded by total params (7.05B × 2 B = 14.1 GB), not by batch size. It is held within budget by (a) **~30× gradient compression** (top-2% sparsification + INT8 quantization + error feedback on expert/shared/router/attention grads, DeepSpeed-Compression / 1-bit Adam style, FP32 masters kept local), shrinking per-node egress to **≈ 0.8 GB ≈ 0.26 s** at 25 Gbps; (b) **gradient accumulation** to a large step batch (8.4M tokens) so compute per step (~0.5 s on 32 GPUs, ~2.1 s on 8) is ≥ the sync window; and (c) **overlapping** the all-reduce with backward pass (gradient bucketing, priority-based co-scheduling so it does not contend with the intra-node all-to-all).

Net inter-node link occupancy ≈ **<10%**, with headroom for traffic spikes; the bottleneck becomes intra-pod all-to-all (~2 s/step), which is cheap, overlapping NVLink traffic.

---

## 4. Routing choice with justification

**Choice: learned softmax top-2 routing (greedy, deterministic) + load-balancing auxiliary loss + capacity factor ≥ 1.0 (no token dropping).**

- **top_k = 2** (vs top-1): top-1 halves intra-pod dispatch volume and is simpler (Llama 4, Switch), but top-2 consistently gives better quality and acts as regularizer (Mixtral; DBRX went further to k=4). Since token traffic is **intra-node and cheap here**, spending 2× on it is free in inter-node terms — k=2 is the proven sweet spot (Mixtral, DeepSeek-V2/V3).
- **k=2 fixed, not k=4+**: higher k raises expert-load variance (hurts a 4-way shard) and inflates combine work, with diminishing quality returns at 1B active scale.
- **E = 32** (not 64–256): with EP=4 and 8 experts/GPU/layer, this keeps total params at 7.0B, which **directly caps the inter-node gradient volume** (the budget driver). Larger E would add capacity but grow the cross-node sync cost; smaller E (8) reduces specialization headroom. 32 also keeps the fine-grained-MoE benefit (each expert is a small, specialized 8.39M FFN).
- **Group-limited by construction**: DeepSeek-V3's group-limited gating exists specifically to remove cross-node all-to-all; here the pod is the group, so the router is a plain top-2 over the pod-local expert set — no global expert coordination, no cross-pod routing metadata.
- **Balance loss**: with only 4 GPUs sharding 32 experts, load imbalance wastes ¼ of the pod's expert compute and unbalances the all-to-all. A small auxiliary balance loss (coefficient ~0.001–0.003, Mixtral/DeepSeek) plus capacity factor ≥1.0 keeps utilization high without dropping tokens.

---

## 5. Training implications

- **Parallelism recipe: EP=4 (intra-pod) × DP=K (across pods); no TP/PP/SP needed** at d=2048 / 24 layers. Add pods to scale throughput, not model size.
- **Intra-node all-to-all becomes the dominant cost** (~2.1 s/step at 200 GB/s vs ~0.5 s compute on 32 GPUs): use DeepEP / Hybrid-EP-style dispatch-combine kernels (NVLink, few SMs, overlap with expert GEMMs), chunked send, and priority-based co-scheduling of all-to-all vs gradient all-reduce to avoid fabric contention (IETF draft-li-moe-ep).
- **Inter-node gradient sync** must stay compressed: top-k (≈2%) + INT8 + error feedback for expert/shared/router/attention grads, FP32 masters, ring all-reduce overlapped with backward; keep per-step batch large (8.4M tokens) to amortize the fixed 14.1 GB sync.
- **Memory/GPU (80 GB class):** weights ≈ 4.8 GB (BF16: attention + shared + router + embedding replicated, 6.44B/4 expert shard ≈ 3.2 GB); optimizer ≈ 29 GB (12 B/param for 2.43B per-GPU params); activations ~5–10 GB with checkpointing. Fits 80 GB; on 40–48 GB parts, enable 8-bit Adam and tighter checkpointing.
- **Load balancing is a first-class training concern** at EP=4: monitor per-expert utilization each step; keep the balance loss live through training (do not drop it during fine-tuning, which causes expert collapse).
- **Stability:** shared expert absorbs common computation and stabilizes routing; use BF16 with FP32 master, and validate that compressed-gradient error feedback does not drift validation loss (roll back to fp16 sync if so).

---

## 6. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Pod spans >1 node** (e.g., 2+2 GPUs) — assumption that a "pod of 4" is one node with fast fabric | Partial all-to-all crosses 25 Gbps → ~tens of s/step, budget violated | Verify topology before build. If it spans nodes: drop to top_k=1, or shrink per-step tokens, or restrict dispatch to intra-node expert pairs and replicate the remaining pair |
| Inter-node gradient sync becomes the floor despite compression | Wall-clock bound; compressed grads can hurt convergence | 1-bit Adam-style error feedback; monitor val-loss; fallback = fp16 sync + larger step batch (fixed bytes amortize) |
| Load imbalance / expert collapse with only 4-way sharding | Idle expert GPUs (≤25% MFU loss), unbalanced all-to-all | Balance loss throughout training; capacity factor ≥1.0; per-expert utilization telemetry |
| Intra-node all-to-all dominates step time (~2 s vs ~0.5 s compute at 32 GPUs) | Throughput capped by NVLink, not FLOPs | DeepEP dispatch/combine kernels, comm/compute overlap, chunked transfer; reduce per-GPU tokens |
| Router instability (top-2 + 32 experts at small scale) | Churn, dead experts, quality dip early | Shared expert; small balance-loss coefficient; warm-up of router temperature; no token dropping |
| **EP=4 fixed** caps future model growth | Scaling capacity requires bigger per-GPU expert shards, not more pods | Raise E (and thus total params/inter-node sync) only when link budget allows; or add intra-pod TP for expert shards |
| Link is 25 Gbps *per node*, possibly shared/shaped | Effective per-GPU bandwidth as low as 6.25 Gbps | Assume egress = one 25 Gbps link/node; keep occupancy <10% so shaping headroom is large |

---

**Bottom line:** EP = 4, with the EP group pinned inside a 4-GPU pod, moves the entire 3.30 TB/step token-communication volume onto intra-node NVLink, leaving only ~0.8 GB/step of compressed gradient traffic on the 25 Gbps inter-node link — ~0.26 s per step against a multi-second step budget. The 1B dense-equivalent target is met exactly (1.01B active params), with 32 experts and top_k = 2.