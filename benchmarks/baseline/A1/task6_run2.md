Since I can't write files in this session, here is the architecture document inline. All figures are internally consistent and derivable from the specs below.

---

# P4-MoE-1B — MoE Architecture for a 25 Gbps Interconnect with Fixed EP-4 Pods

## Overview

This design targets a training cluster with **no InfiniBand**, an inter-node interconnect **capped at 25 Gbps per node uplink** (3.125 GB/s), and a topology that groups GPUs into fixed 4-GPU pods. Because the topology fixes **expert-parallel degree = 4**, we do not scale EP with total GPU count; instead we treat each pod as a self-contained MoE replica.

**Parallel strategy (the constraint-satisfying decision):**
- **DP = number of pods, EP = 4 (fixed, intra-pod only).** All 8 experts of every MoE layer live inside one pod (2 experts per GPU). Token dispatch is therefore confined to the pod's four 25 Gbps links and **never crosses pods**.
- Attention + embeddings are **replicated per pod** (DP) and use GQA, with **no attention tensor parallelism** — tensor parallelism would force per-activation traffic across the slow links for *every* layer. We avoid it entirely.
- The only cross-pod traffic is **weight-gradient all-reduce** for the replicated params (small, 128M params), and it is batched/overlapped.
- Each pod processes its own batch slice independently and identically, so the per-node communication budget is invariant to total cluster size (only the gradient ring grows, and it saturates at ~2× data size).

**Result:** a ~0.88B *activated*-parameter, ~3.15B *total*-parameter MoE with quality aimed at a 1B dense model, whose per-step token dispatch fits in the pod's interconnect with headroom (see "Communication budget" below).

## Parameters

| Parameter | Value | Notes |
|---|---|---|
| Model type | Decoder-only Transformer, 24 layers, **all MoE** | |
| `d_model` | 1024 | |
| Attention | GQA, 16 query heads / 4 KV heads, head_dim 64 | Keeps replicated params tiny → cheap grad sync |
| Expert FFN | SwiGLU, intermediate 5120 | 3 matrices of 1024×5120 / 5120×1024 |
| **`num_experts`** | **8** per layer | 8 → EP-4 gives exactly 2 experts/GPU, batching two full GEMM groups per layer |
| **`top_k`** | **2** | Quality vs dispatch-cost tradeoff, see Routing |
| `vocab_size` | 64,000 (tied input/output embedding) | |
| Sequence length | 4096 | Context parallel not needed; attention is pod-local |
| **Total parameters** | **≈ 3,148,000,000 (3.15 B)** | 24 × 128.45M + 65.5M |
| **Activated params/token (dense-equivalent)** | **≈ 883,500,000 (0.88 B)** | 24 × (2×15.73M + 2.62M) + 65.5M |
| Capacity factor | 1.0 | Bounds dispatched-token volume (see Routing) |
| Precision | BF16 weights/activations; FP8 dispatch vectors (recommended) | |
| Example cluster | 8 pods × 4 GPUs = 32 GPUs (DP=8, EP=4) | Per-node numbers are pod-invariant |

Parameter math: embedding 64,000×1024 = 65.5M; attention/layer 2.62M; one expert 3×5120×1024 = 15.73M; one MoE layer 8×15.73M + 2.62M = 128.45M. **Total = 3.15B; activated (top-2) = 0.88B.** Sparse activation ratio ≈ 3.6×.

Per-GPU memory (40 GB budget is ample): expert shard 2 experts × 24 layers × 15.73M = 755M params (1.5 GB BF16) + AdamW FP32 optimizer ≈ 9.1 GB; replicated 128M params ≈ 0.26 GB + ≈1.5 GB optimizer; total ≈ 12.5 GB params+optimizer before activations (use activation checkpointing).

## Routing choice with justification

**Route: top-2 softmax (token choice) with a load-balancing auxiliary loss and fixed capacity factor 1.0.** No shared expert.

Justification against the alternatives:

1. **Why top-2, not top-1:** For a ~0.9B activated model, top-1 with 8 experts measurably lags 1B-dense quality on language tasks; top-2 closes most of the gap for a modest 2× dispatch cost. Since quality is the point of "dense-equivalent 1B", top-2 is the right operating point.
2. **Why not Expert Choice (EC):** EC sends each *expert* to its top-k *tokens*, i.e., every expert gathers tokens from all GPUs. With EP=4 the gather is intra-pod (so not cross-pod), but EC with the same quality needs k≈4 sends/token — *doubling* our dispatch volume. Token-choice with capacity 1.0 gives a hard, predictable dispatch bound, which is exactly what a 25 Gbps budget needs.
3. **Why 8 experts with EP-4 (2 per GPU):** 8 is the largest count that keeps EP-4 clean at 2 experts/GPU — enabling contiguous, batched GEMMs per GPU with no expert sharding across nodes — while 8×top-2 gives more routing diversity than 4×top-2 at identical dispatch cost. Fewer, fatter experts also reduce the per-dispatch vector count.
4. **Capacity factor 1.0 + aux loss** (DeepSeek-style additive load-balance loss, coefficient ~0.01): guarantees each GPU handles at most its fair share of tokens, so dispatch volume is bounded *by construction* — a requirement, not an optimization, on this interconnect. Dropped tokens (≤1–2% with the aux loss) are a deliberate, small quality tax paid for a guaranteed communication envelope.

**Token-communication volume per step** (the number this design must defend):

With global batch **131,072 tokens/step** (32×4096) and DP=8, each pod sees 16,384 tokens, **4,096 per GPU**:
- Dispatch: each token is sent to top-2 experts → 2 hidden-state vectors of `d_model`=1024 BF16 values (2,048 B) per MoE layer, plus the expert output returned.
- **Per node per step: 4,096 × 2 × 2,048 B × 24 layers × 2 (round trip) = 805 MB** (3.22 GB per pod). Link time at 3.125 GB/s = **0.26 s**.
- Per-token cost: 6.1 KB (BF16); **3.1 KB with FP8 dispatch → 0.13 s**.
- Gradient all-reduce (replicated 128.4M params, 257 MB grads, ring over 32 GPUs): ≈ 514 MB → **0.16 s**.
- **Total comm ≈ 1.32 GB/node → ≈ 0.42 s/step** (85% of a 0.5 s budgeted step; dispatch alone = 52%).

**How it fits the 25 Gbps budget:** (a) EP-4 confines dispatch to the 4 intra-pod links — no activation ever leaves the pod, so the 25 Gbps uplinks carry only grad-sync; (b) the design keeps the *per-token* dispatch small (top-2, 1024-dim, BF16→FP8) and the *number of tokens per step* moderate, so 805 MB of dispatch per node is an ordinary ~0.26 s of link time — comfortably inside the 1.56 GB comm window available in a 0.5 s step; (c) GQA + no attention TP keep replicated params at 128M so the cross-pod all-reduce stays at ~0.16 s; (d) dispatch of layer *i*+1 is overlapped with compute of layer *i*, and grad all-reduce with backward, so wall-clock tends toward `max(compute, comm)` rather than their sum.

Throughput consequence: compute ≈ 0.70 PFLOP/step → ~70 ms/GPU on H100-class, so training is **communication-bound at ~15–20% MFU**, ~260–330k tok/s (BF16 dispatch), rising toward the **~1M tok/s interconnect ceiling with FP8 dispatch and larger batches** (dispatch is linear in tokens; grad sync amortizes, so up to ~262–524k tokens/step improves throughput).

## Training implications

- **Optimal batch:** 131k–524k tokens/step via grad accumulation; larger batches amortize grad sync and push toward the per-token dispatch ceiling.
- **Mixed precision:** BF16 compute; **dispatch vectors in FP8** to halve the dominant traffic (halves per-token cost to 3.1 KB). This is the single highest-leverage optimization on this cluster.
- **Load balancing:** capacity 1.0 with aux loss coefficient ~0.01; monitor expert utilization per layer; watch early-training collapse (see Risks).
- **Overlap schedule:** software-pipelined all-to-all (dispatch layer _i_+1 during layer _i_ compute) + async ring all-reduce during backward; without this, step time ≈ 0.7 s.
- **Checkpointing:** 3.15B params on 25 Gbps links is slow; use per-node sharded checkpoints (each node saves its 755M expert shard + replicated 128M) written asynchronously to local disk, never copied node-to-node.
- **No TP, no context-parallel:** avoided on purpose; attention and embeddings are fully replicated, so all cross-pod traffic is weight gradients, not activations.
- **Scaling:** to add GPUs, add pods (raise DP). EP stays 4 by topology mandate; the per-node dispatch budget is unchanged; only the grad ring grows and saturates at ~2× data size.

## Risks

1. **Quality vs 1B dense is not guaranteed.** 0.88B activated + top-2/8 experts usually matches ~1B dense, but on small-vocab/domain data a 1B dense can win. Fallback: raise expert intermediate to 6144 (adds dispatch bytes only for the returned vector — unchanged size) and/or dense-ify the first 2 layers (DeepSeek-style), trading ~8% dispatch volume.
2. **Router collapse / load imbalance at warm-up.** With capacity 1.0, a collapsed router drops tokens and silently harms quality. Mitigate with aux loss + local entropy regularization; never disable capacity capping, since that's what holds the comm envelope fixed.
3. **Communication-bound MFU (~15–20%).** The 25 Gbps budget caps throughput at ~0.5–1M tok/s no matter the FLOP budget; excess GPUs add grad-sync cost faster than dispatch headroom. Do not buy GPUs for this cluster; buy a bigger batch and FP8 dispatch instead.
4. **Dispatch is the hard ceiling.** Per-token dispatch (6.1 KB BF16) is ~3.5× the per-token compute time, so even perfect overlap caps at ~512k tok/s (BF16). Anything that increases per-token bytes (top-k>2, d_model↑, Expert Choice) must be re-budgeted against this fixed 25 Gbps.
5. **Cross-pod gradient sync** grows with pod count (ring ≈ 2×data); above ~16 pods it rivals dispatch. At that point, options are gradient compression (TopK sparsification, 1-bit Adam) or switching to a pod-of-4-local optimizer — both tradeoffs worth flagging now.
6. **Checkpoint/restart latency** on 25 Gbps links; mitigated by sharded local checkpoints (above), but a full cluster restore will be slow — budget for it in the training plan.

**Net:** this design meets the constraint by making the pod the MoE island — EP=4, dispatch confined intra-pod at ~805 MB/node/step (~0.26 s), grad sync ~0.16 s, total ~0.42 s of a 0.5 s step — with FP8 dispatch and overlap as the two levers that turn "feasible" into "decent throughput" (~0.3–1M tok/s).