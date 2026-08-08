# MoE Architecture: 1B Dense-Equivalent on a 25 Gbps, Pod-of-4 Cluster

## 1. Overview

The target is a **1B dense-equivalent** MoE that must train on a cluster whose **inter-node interconnect is capped at 25 Gbps** (no InfiniBand) and whose topology **fixes expert-parallel degree at EP = 4** (GPUs grouped into fixed pods of 4). The dense baseline is 16 layers, `d_model` 2048, FFN 8192 (`ffn_mult` 4), vocab 32,000, seq_len 2048 → **1.14B parameters**. The MoE replaces each dense FFN with **8 experts of width 4,096 (half the dense FFN)** under **top-2** routing. Because each expert is half the dense FFN and two are active, **activated params (1.14B) equal the dense baseline** (compute-equivalent), while **total params grow to 3.56B (3.12× ratio)**. 

The constraint is satisfied structurally: expert parallelism is **fixed at EP = 4**, and each EP group is placed **entirely inside one 4-GPU pod**, so the token all-to-all — **17.18 GB per MoE layer, 274.9 GB per step** at a 1,048,576-token global batch — rides the intra-pod fabric and **never touches the 25 Gbps inter-node link**. The only inter-node traffic is the data-parallel gradient all-reduce of the replicated (non-expert) parameters: **~1.0 GB per step ≈ 0.32 s ≈ 6% of a ~5 s step**, leaving ≥15× headroom on the 25 Gbps budget.

## 2. Parameters table

**`num_experts` = 8**, **`top_k` = 2**, expert `ffn_dim` = 4,096, `num_layers` = 16, `d_model` = 2048, vocab = 32,000, capacity factor = 1.0. All figures follow the `moe_calculator.py` formulas.

| Component | Dense | MoE | Activated |
| --- | ---: | ---: | ---: |
| Attention (all layers) | 268,435,456 | 268,435,456 | 268,435,456 |
| Expert FFNs (all layers) | 805,306,368 | 3,221,225,472 | 805,306,368 |
| LayerNorms | 65,536 | 65,536 | 65,536 |
| Embedding | 65,536,000 | 65,536,000 | 65,536,000 |
| Router | — | 262,144 | 262,144 |
| **Total** | **1,139,343,360** | **3,555,524,608** | **1,139,605,504** |

- Total (MoE) = **3,555,524,608 ≈ 3.56B parameters**
- Activated = **1,139,605,504 ≈ 1.14B** (equals the dense baseline → identical FLOPs/token)
- **Param ratio = 3.12×** (Mixtral-like; the router's 262,144 is <0.01% and shown for completeness)
- FLOPs/token = 6·1.14B + 4·16·2048·2048 ≈ **7.10 GFLOP** — the MoE trains at dense-equivalent compute

## 3. Routing choice

| Decision | Value | Justification |
| --- | --- | --- |
| Strategy / top-k | **Top-2** | Training-quality sweet spot; at this scale the quality gain over Top-1 exceeds the +1 dispatch per token. |
| Capacity factor | **1.0** | Minimizes padded dispatch buffers and message volume — directly serves the 25 Gbps constraint. |
| Aux loss | **0.01** | Upper end of the range: with cf = 1.0 we tolerate no imbalance (drops would lose tokens), so load-balancing is deliberately strong. |
| Router jitter | On | Breaks deterministic early specialization; keeps all 8 experts fed so no single EP rank becomes a pod straggler. |

Note: expert FFN width 4,096 = dense FFN ÷ top_k gives **total expert capacity 4× the dense FFN** (matching the "2–4× expert capacity" rule) while keeping activated capacity compute-neutral.

## 4. Training implications

**Compute.** 7.10 GFLOP/token, identical to the dense baseline. On 16 GPUs (4 pods × 4, A100-class ~150 effective TFLOPS → ~2.4 PFLOPS), peak ~338K tokens/s, realistic ~200K tokens/s (~60% MFU); a 1,048,576-token optimizer step ≈ 5 s.

**Memory per GPU** (bf16 + AdamW mixed, EP=4, micro-batch 8×2048, activations recomputed): params 2.28 GB + gradients 2.28 GB + optimizer 13.68 GB (12 B/param) + activations 5.37 GB + expert buffers ~0.5 GB ≈ **26.6 GB** → ~33% headroom on 40 GB GPUs. Params resident per rank = activated params (replicated dense part + EP-sharded experts), so the estimator's DP/EP sharding is consistent.

**Parallelism.** `DP × EP = 4 × 4 = 16 GPUs` (generalizes to `DP = GPUs/4`); TP=PP=1 (avoided to cut collectives). EP = 4 divides `num_experts` = 8 ✓. Tokens per expert per GPU per micro-step = 8·2048·2/(8/4) = **16,384**, far above the 8–64 utilization floor.

**Constraint: token-communication volume per step vs 25 Gbps.** At the 1,048,576-token global batch, the expert all-to-all per step is

`2 directions × top_k(2) × 1,048,576 tokens × d_model(2048) × 2 B = 17,179,869,184 B = 17.18 GB per MoE layer`, × 16 MoE layers = **274.9 GB per step**.

If this crossed the 25 Gbps link it would take **~88 s/step — infeasible**. The design therefore never routes it there: **EP = 4 confines each all-to-all to one 4-GPU pod** (68.7 GB/pod/step over NVLink, ~0.17 s), so the 25 Gbps inter-node budget is consumed only by the DP gradient all-reduce of replicated params (attention + LN + embedding + router = 334,299,136 B → 0.67 GB raw, ×1.5 ring factor ≈ **1.0 GB/step ≈ 0.32 s ≈ 1.6 Gbps ≈ 6% of the 25 Gbps budget**, ≥15× headroom). Top-k, cf, and expert count are tuned to keep the intra-pod all-to-all cheap and the inter-node share negligible.

## 5. Risks

1. **Topology assumption is load-bearing.** The 274.9 GB/step all-to-all is only safe if every EP group sits in one pod. If node assignment ever straddles pods, inter-node traffic jumps to ~88 s/step. **Mitigation:** pin EP groups to pods at launch and verify `torch.distributed` rank-to-node mapping before training.
2. **Token drops at cf = 1.0.** Any imbalance drops tokens and silently changes the effective batch. **Mitigation:** aux 0.01 + jitter; monitor effective experts and drop rate; fall back to cf = 1.25 if drops exceed ~1%.
3. **DP gradient sync on the critical path.** ~6% of step time on a slow link; worse if step time falls below ~3 s. **Mitigation:** overlap sync with backward; later, gradient compression (TopK/QSGD).
4. **Over-parameterization vs compute.** 3.56B total for 1B compute makes the optimizer the memory driver (13.7 GB). **Mitigation:** fits with headroom; if GPUs are 32 GB, switch to 8-bit AdamW or raise EP is unavailable (EP is fixed) → lower micro-batch instead.
5. **Router collapse under strong aux.** Aux 0.01 can distort routing. **Mitigation:** keep it at the stated 0.01, confirm via effective-expert count; do not push higher.