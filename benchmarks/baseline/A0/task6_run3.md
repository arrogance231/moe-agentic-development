# MoE Architecture for a 25 Gbps / Fixed EP-4 Cluster (1B Dense-Equivalent)

## Overview

This design targets a 1B dense-equivalent MoE LM trained on a cluster whose defining constraints are:

- **Inter-node interconnect: 25 Gbps Ethernet** (no InfiniBand/RoCE assumed).
- **Expert-parallel degree fixed at `EP = 4`**, because the topology groups GPUs into fixed pods of 4 nodes; this holds regardless of total GPU count.

The model is a Mixtral-style **token-choice top-2 MoE** scaled to ~1.0B parameters. The load-bearing insight is that **expert parallelism is confined inside one 4-node pod**: all token dispatch/combine traffic crosses only the pod fabric, and it scales linearly with tokens — exactly like compute — so its fraction of the 25 Gbps budget is constant (~4%) no matter the batch size. Scaling out is done with **data parallelism across pods**, which adds only the (overlappable) gradient all-reduce to the interconnect.

No single published recipe combines a 1B top-2 MoE with a small fixed EP and a 25 Gbps fabric, so this document states the design explicitly.

---

## Parameters

| Parameter | Value |
|---|---|
| Vocabulary size | 32,768 |
| `d_model` | 1,024 |
| Num layers | 26 |
| **`num_experts`** | **8** |
| **`top_k`** | **2** |
| Expert FFN intermediate | 2,048 |
| Precision | bf16 (2 bytes/param) |
| Token/sequence | 4,096-token sequences |

| Component | Formula | Parameters |
|---|---|---|
| Token embedding | 32,768 × 1,024 | 33,554,432 |
| Attention (all layers) | 26 × 4 × 1,024² | 109,051,904 |
| Experts (all layers) | 26 × 8 × (1,024·2,048 + 2,048·1,024) | 872,415,232 |
| Routers | 26 × 1,024 × 8 | 212,992 |
| **Total** | | **1,015,234,560 (~1.02B)** |
| Active params/token | embed + attn + (2/8)·experts | ~360.7M |

Compute per token ≈ 2 × 0.36G ≈ 0.7 GFLOPs → ~2.8× cheaper than a 1B dense forward, while retaining 8-expert capacity.

---

## Routing choice

**Token-choice, `top_k = 2`, softmax top-2 (Mixtral-style) + auxiliary load-balancing loss (λ ≈ 0.01) and a capacity factor of 1.0–1.25.**

Justification against alternatives:

- **top-1 (Switch/ST-MoE):** halves dispatch traffic (1 hop/token) but has a well-documented quality gap. Since our all-to-all uses only ~4% of the interconnect budget (below), the bandwidth savings are not worth the quality loss.
- **Expert-choice routing:** guarantees balanced compute and would remove straggler risk on asymmetric Ethernet, but adds algorithmic complexity. We stay with token-choice + aux loss + capacity factor, and keep expert-choice as the documented fallback if imbalance appears in practice (see Risks).
- **Fine-grained + shared experts (DeepSeek-V2/V3 style):** increases dispatch volume per token and expert count; unnecessary at 1B scale and on a small pod.
- **`E = 8, EP = 4` ⇒ 2 experts per node.** This is the minimum expert count that keeps the off-node dispatch fraction at the fixed 3/4 per routed hop while giving enough capacity; `EP = E` (one expert/node) is strictly worse (all hops off-node, no capacity headroom).

Routing math is local per GPU (router is replicated); only the small token→expert index counts are exchanged (~KB-scale), not the router parameters.

---

## Training implications

**Expert-parallel degree used: `EP = 4`** — experts 0–3 on nodes A–D of each 4-node pod; each node holds 2 experts. No expert communication leaves the pod.

**Token-communication volume per step (per EP group / pod):**

Per token, in bf16, each hidden-state vector = `d_model × 2 = 2,048 B`. With `top_k = 2` and 4 nodes, the expected number of off-node routed hops per token is `2 × 3/4 = 1.5`. Four all-to-all passes occur per step (dispatch fwd, combine fwd, dispatch bwd, combine bwd):

```
per token:      4 passes × 1.5 hops × 2,048 B  = 12 KB
per step (65,536 tokens/pod, e.g. 16×4096, grad-accum 4):
                65,536 × 12 KB                  = 786 MB all-to-all (0.75 GB)
per node egress: 786 MB / 4                     = ~197 MB
```

**Budget check — all-to-all:**

```
25 Gbps = 3.125 GB/s
all-to-all time/node  = 197 MB / 3.125 GB/s  ≈ 0.063 s
per-step compute on pod (4 GPUs ≈ 240 TFLOP/s, 65,536 tok) ≈ 1.66 s
→ interconnect occupancy ≈ 4%  ⇒ 96% headroom
```

Because both all-to-all volume and compute scale linearly with tokens per pod, **this ~4% occupancy is invariant to batch size** — the design never saturates the 25 Gbps fabric via expert traffic.

**Budget check — data-parallel gradient all-reduce (cross-pod):**

The only other fabric traffic is the DP gradient all-reduce, sized by the model, not by tokens:

```
grad size/replica          = 1.02B × 2 B        = 2.03 GB
ring all-reduce, 16 pods   = 2 × (15/16) × 2.03 ≈ 1.90 GB/node → ≈ 0.61 s on 25 Gbps
per-step compute, 64 GPUs, 1M global tokens    ≈ 1.67 s
```

This 0.61 s is hidden by overlapping all-reduce with backprop (grad chunking / wait-free buckets). It becomes the *true* floor only at extreme DP counts with tiny steps; mitigate via fewer grad-accum steps or bf16-grad compression (Risks).

Operational choices:

- **Communication–compute overlap:** chunk the batch and overlap the all-to-all of chunk *i+1* with expert GEMMs of chunk *i*; jumbo frames / MTU 9000 since plain Ethernet (no RoCE) makes message count and loss the real enemies.
- **Grouped GEMMs:** each node executes its 2 experts with grouped GEMM (no per-expert kernel launch overhead).
- **DP across pods** with independent optimizer state; standard ZeRO-sharded optimizer within a pod.
- bf16 throughout; no fp32 master except optimizer state.

---

## Risks

1. **Straggler/expert imbalance (highest).** Token-choice routing on asymmetric Ethernet can idle 25% of expert compute; the all-to-all is synchronous, so imbalance converts directly to latency, not just throughput loss. *Mitigation:* aux load-balancing loss, capacity factor 1.25, expert-choice routing as fallback.
2. **DP all-reduce is the real bandwidth ceiling.** 2.03 GB/replica across a 25 Gbps fabric sets a ~0.6 s/step floor that does *not* shrink with batch. *Mitigation:* overlap with backprop, reduce per-step tokens at high DP, optional gradient compression.
3. **Plain-Ethernet loss/latency.** 25 Gbps Ethernet without RoCE means TCP retransmits and RTT dominate if messages are small or bursty. *Mitigation:* coalesce dispatch into large messages, MTU 9000, chunked/overlapped all-to-all.
4. **Capacity capped by fixed EP=4.** Expert count is limited to what fits 4 nodes (here `E=8`, 2/node); total capacity can only grow via data-parallel replicas or by raising `E` to 16 (4/node), which raises memory but not dispatch volume per token.
5. **Routing quality sensitivity.** Aux-loss weight and capacity factor must be tuned; too high a capacity factor allows collapse toward dense behavior, too low causes dropped tokens.
6. **Scaling ceiling.** EP cannot track GPU count, so beyond ~4 pods the design is DP-bound by Risk 2 rather than expert-parallel-bound — an intentional trade-off forced by the pod topology.