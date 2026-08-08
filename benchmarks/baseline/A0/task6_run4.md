I don't have file tools in this session, so I'll deliver the architecture document directly. Here it is.

---

# MoE Architecture for a 25 Gbps Ethernet Cluster with Fixed Expert-Parallel Degree 4

**Constraint recap:** inter-node interconnect capped at 25 Gbps (no InfiniBand); GPUs are fixed in pods of 4, so **expert-parallel (EP) degree = 4 regardless of total GPU count**; target dense-equivalent size = 1B parameters. No published MoE recipe (ST-MoE, Mixtral-8x7B, Switch, DeepSeek-V3) assumes this combination — all assume high-bandwidth (NVLink/IB) expert transport and scale EP with expert count.

## Overview

We design a top-2-of-8 Mixture-of-Experts transformer whose *active* parameter count per token is **~0.97B**, i.e. a 1B dense-equivalent. The key structural decision forced by the topology is that **one EP group = one pod of 4 GPUs, and all 8 experts are placed inside the pod (2 experts per GPU)**. Expert dispatch/combine traffic therefore stays *intra-pod* and never crosses the 25 Gbps inter-node link by design. The 25 Gbps fabric is only traversed by **data-parallel gradient all-reduce**, whose volume we bound and amortize explicitly.

Scale-out is data-parallel only: with a total of `G` GPUs, `D = G / 4` pods each hold a full model replica and synchronize gradients over the 25 Gbps fabric. For `G = 64` (16 pods) the design is worked out below.

## Parameters

| Item | Value |
|---|---|
| `d_model` | 2048 |
| Transformer layers | 18 |
| Attention heads × head_dim | 16 × 128 |
| **num_experts** | **8** |
| **top_k** | **2** |
| Expert FFN dim (hidden) | 4096 |
| Vocab size | 32,768 |
| Shared params (embedding + 18× attention + gate) | ~369 M |
| Expert params (18 × 8 experts) | ~2,416 M |
| **Total parameters** | **~2,785 M (~2.79B)** |
| **Active params / token (dense-equivalent)** | **~0.97B ≈ 1B** |
| **Expert-parallel degree (fixed by topology)** | **4** |
| Data-parallel degree | `total_GPUs / 4` (16 at 64 GPUs) |

Derivation: per-layer attention = `4·d² = 16.78M`; per-expert MLP = `2·2048·4096 = 16.78M`, so 8 experts/layer = `134.2M`. Active/token = `67.1M (emb) + 302M (attn) + 18 × 2 × 16.78M (top-2 experts) = 973M ≈ 0.97B`. This matches the "1B dense-equivalent" target in the standard sense: the same per-token compute as a ~1B dense model, at 2.8× total parameters from sparsity.

## Routing choice (with justification)

**Softmax top-2 gating with an auxiliary load-balancing loss (`aux_loss ≈ 0.01`) and no token-drop (capacity factor 1.0 + balanced dispatch).**

- **Why top-2, not top-1 (Switch):** top-2 gives better expert specialization and more stable training than top-1 for the same expert count, and it keeps the token fan-out (and hence all-to-all volume) modest — doubling top_k from 1→2 only doubles comm, which we have ample budget for.
- **Why aux-loss-balanced routing, not Expert-Choice:** load-balancing loss keeps per-device token counts uniform, which is what makes the communication volume *predictable* and lets us certify it against the 25 Gbps budget. Expert-Choice (perfect balance) changes the comm pattern to larger, skewed all-to-all tensors and adds complexity we don't need.
- **Why capacity factor 1.0:** with a balanced router the capacity is matched to the fixed 2-expert-per-GPU allocation; dropping tokens (the Switch failure mode) is avoided to keep training loss-smooth on a budget-constrained cluster where every restart is expensive.

## Training implications

**Expert all-to-all (the token communication) — kept within the 25 Gbps budget:**

Per device, per layer, the two all-to-all phases (dispatch + combine) move:

```
V_layer = 2 × B × (top_k × (1 − E_per_device / E)) × d_model × 2 bytes
        = 2 × 2048 × (2 × (1 − 2/8)) × 2048 × 2
        = 25.2 MB / device / layer
```

With 18 layers and `B = 2048` tokens/device/step:

| Quantity | Value |
|---|---|
| **Token-comm volume / step / device** | **18 × 25.2 MB = 453 MB (~110k token-vector transmissions)** |
| **Token-comm volume / step / pod (4 GPUs)** | **1.81 GB** |
| Token-comm volume / step / cluster (64 GPUs) | 29.0 GB aggregate |

**Budget check.** 25 Gbps = 3.125 GB/s. Even under the *pessimistic* assumption that all expert traffic crosses a single 25 Gbps link: per-layer per-device 25.2 MB → **8.1 ms**, per-step per-device 453 MB → **145 ms**, versus ~40–60 ms of per-layer compute (≈4 TFLOPs/layer/device at ~100 TFLOPs) — comm is fully hidden behind compute with >70% link headroom. In reality the assumption is moot: **all 8 experts are resident in the 4-GPU pod, so dispatch/combine is intra-pod fabric, and the 25 Gbps link carries zero expert traffic.**

**The real 25 Gbps traffic is gradient all-reduce.** At `D = 16` pods, grads = 2.79B × 2B = 5.58 GB/step; ring all-reduce moves ≈`(D−1)/D × 5.58 ≈ 5.2 GB` per link → **~1.7 s/step on 25 Gbps**. Mitigations: (a) overlap all-reduce with backward compute; (b) 8-bit gradient compression → halves to ~0.85 s; (c) larger global batch (raise `B` to 4k–8k tokens/device) so gradient sync is amortized over more tokens. With these, target step time ~2–3 s is achievable.

**Practical notes:** per-GPU expert memory is tiny (2 experts × 16.78M × 2B × 18 layers ≈ 1.2 GB), so memory is not a constraint; pure DP across pods is chosen over pipeline or tensor parallelism because the model is small and pod-bounded parallelism already covers experts; use bf16 activations/weights throughout; checkpoint (~5.6 GB) saves in ~1.8 s, acceptable.

## Risks

- **Load imbalance breaks the budget guarantee.** If the aux-loss is mis-tuned, hot experts skew the all-to-all and inflate both latency and idle time. *Mitigation:* monitor per-expert utilization; tune `aux_loss` coef; fall back to capacity-buffered dispatch before resorting to token-drop.
- **Gradient sync is the scaling bottleneck, not expert comm.** At large `D`, the 5.2 GB/step all-reduce on 25 Gbps dominates (≈1.7 s/step at 16 pods). *Mitigation:* 8-bit compression, computation/communication overlap, larger batch. This is the constraint the design must re-validate at every cluster scale-up.
- **EP=4 caps expert capacity per pod, and growth is awkward.** All 8 experts must live on 4 GPUs; raising `num_experts` (to e.g. 16) without increasing EP means more experts per GPU (fine for memory, but per-expert compute per token shrinks). Conversely, if a future model ever needs EP > 4, expert traffic *will* cross the 25 Gbps fabric and immediately violate the budget. *Mitigation:* keep `num_experts ≤ 8` and grow by data parallelism only.
- **Fixed topology means no NVLink-style repair if intra-pod fabric degrades.** The design depends on the pod's internal fabric being fast; a half-functional pod silently forces expert traffic onto 25 Gbps links. *Mitigation:* per-pod health checks and topology-aware job scheduling (never split an EP group across pods).

---

One structural note: this session had no file tools, so I couldn't save the document to disk — paste it into a `.md` file (e.g. alongside your other configs) and I can extend it if you want.