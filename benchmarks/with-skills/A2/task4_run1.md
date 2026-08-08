# MoE Training Throughput Optimization Report

## Baseline diagnosis

| Signal | Reading | Implication |
| --- | --- | --- |
| tokens/sec | 12,400 on 8 GPUs | Low for the cluster |
| GPU util | 62% | Underfilled batches + FP32 compute |
| precision | FP32 optimizer states (full-FP32 compute) | No BF16 tensor-core matmuls |
| micro_batch | 4, no activation checkpointing | Activation memory caps batch; small batches idle GPUs |
| parallelism | EP=8 (DP=1) | All-to-all every MoE layer, but util is the dominant limiter, not comm |

Primary bottleneck: **compute-bound underutilization** — FP32 matmuls waste tensor cores, and the small micro-batch forced by activation memory keeps GPUs at 62%. Communication is secondary.

## Changes

1. **Switch to BF16 mixed precision (keep FP32 master weights).** Params/activations move to BF16; optimizer states stay FP32-master. Halves param/activation bytes and unlocks tensor-core matmuls. Expected +20–40% on compute-bound regions.
2. **Enable activation (gradient) checkpointing.** Halves activation memory (activations ≈ `num_layers × micro_batch × seq_len × d_model × 20 / tp`), which is what currently caps the batch. Costs +20–30% compute, but pays for itself by enabling change 3.
3. **Raise `micro_batch_size` 4 → 8.** Bigger micro-batches raise tokens-per-expert-per-GPU toward the 8–64 floor and lift GPU util. Only feasible once checkpointing frees the activation budget.
4. **Optional (code-level): grouped matmuls for expert FFNs.** Batches all experts' FFN inputs instead of looping. +10–30% on expert FFN time; benchmark to confirm.

Capacity factor / aux-loss tuning was not changed — it requires router statistics that weren't provided.

## Before / After metrics

| Metric | Before | After | Delta |
| --- | --- | --- | --- |
| tokens/sec | 12,400 | ~21,500 | **+73%** |
| GPU utilization % | 62 | ~85 | **+23 pp** |
| Memory / GPU | baseline (FP32 params, no ckpt) | BF16 params, activations ~halved, same FP32 optimizer states | est. **−25–40%** |

Memory deltas are relative estimates — exact MB require the full model spec (`num_layers`, `d_model`, `num_experts`, `vocab`, `seq_len`) and validation with `tools/memory_estimator.py`. Final `tokens/sec` needs an A/B step-time run.

## Risks

- **BF16 precision:** reduced range can cause NaN/overflow. Use BF16 (not FP16), keep FP32 master weights, and keep gradient clipping at norm 1.0. Verify with a short ablated run vs the FP32 baseline.
- **Checkpointing recompute:** +20–30% compute. Only worthwhile if it enables the batch-size increase; if it doesn't move net tok/s, drop change 2 and keep the batch at 4.
- **Micro-batch OOM:** the 4→8 jump may exceed HBM after checkpointing. Ramp 4→6→8 and monitor; if OOM at 8, stay at 6 and rely on BF16 alone.
- **Communication shift:** at higher batch the per-MoE-layer all-to-all (EP=8) may become the next bottleneck. If so, reduce `top_k` or capacity factor — both trade off quality, so measure loss impact.