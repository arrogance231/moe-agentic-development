Note: I don't have file-write or shell tools in this session, so here's the report inline.

# MoE Training Throughput Optimization Report

## Changes

**1. Parallelism reconfiguration: `expert_parallel=8 → expert_parallel=2, data_parallel=4`**
With EP=8 on 8 GPUs you have zero data/tensor/pipeline parallelism — every token crosses an 8-GPU all-to-all, which dominates runtime and explains the 62% utilization ceiling. Reducing to EP=2 cuts all-to-all communication volume ~4x and gives 4 independent DP streams that can overlap one stream's all-to-all with another's compute. (On single-node NVLink, EP=2/DP=4 typically beats EP=4/DP=2.)

**2. Enable selective activation checkpointing**
Checkpoint only the heaviest activations (MoE expert FFNs + attention), not every layer. Frees large activation memory, enabling a bigger micro-batch at modest recompute cost (~5–10%).

**3. Increase `micro_batch_size: 4 → 16`**
The main lever on utilization is arithmetic intensity. Freed memory from (2) plus (4) lets the batch grow 4x, amortizing all-to-all and kernel-launch overhead per token.

**4. Mixed precision: FP32 → BF16 compute + BF16 optimizer states**
Keep FP32 master weights but store Adam moments and gradients in BF16. Cuts optimizer memory roughly in half, further freeing headroom for the larger batch; BF16 is native on A100/H100.

**5. Overlap all-to-all with compute (fused MoE)**
Use a fused dispatch/combine + grouped-GEMM MoE kernel (e.g., Megatron fused MoE, Transformer Engine) so expert FFNs for earlier layers execute while the next all-to-all is in flight. Add sequence parallelism for attention to shrink activation footprint.

**6. Routing stability: top-2 routing + load-balancing auxiliary loss, capacity factor ~1.1**
Prevents expert load imbalance that silently drops tokens and wastes utilization at higher batch sizes.

## Before/After Metrics

| Metric | Before | After (target) | Delta |
|---|---|---|---|
| Throughput (tokens/sec) | 12,400 | ~21,000 | **+69%** |
| GPU utilization | 62% | ~88% | +26 pts |
| Memory/GPU (activations) | High (no checkpointing) | Recompute on ~50% of layers, but batch up 4x | Net: within budget |
| Memory/GPU (optimizer) | 2×FP32 states + FP32 master | BF16 states + FP32 master | ~−50% optimizer memory |

Roughly: 12,400 × (0.88/0.62) ≈ 17.6k from utilization alone, plus batch-size and communication gains push it to ~21k.

## Risks

- **BF16 optimizer states**: precision loss can hurt convergence/accuracy. Mitigate with FP32 master weights, BF16→FP32 accumulation, and loss scaling; validate loss curves on a short run first.
- **Micro-batch 16 + checkpointing**: OOM risk if the checkpointing is too selective — start at 8, profile, then step to 16.
- **Recompute overhead**: aggressive checkpointing adds compute that erodes the utilization gain; keep it selective.
- **Load imbalance** at higher batch sizes: without the capacity factor/aux loss, uneven expert routing causes idle GPUs and drops token throughput.
- **Communication sensitivity**: overlap gains depend on NVLink topology and NCCL all-to-all tuning; results may differ on multi-node/PCIe setups.
- **Throughput deltas are estimates** — tuning (batch, EP/DP ratio, capacity factor) should be measured, not assumed.