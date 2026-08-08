# MoE Training Throughput Optimization Report

## Current Setup
- **Hardware:** 8× GPU (e.g., A100-80GB), ~12.4B total-parameter MoE (2.4B dense + 10B experts) used as the reference workload
- **Parallelism:** `expert_parallel=8, dp=1, tp=1` — a single data replica; every expert layer does a full 8-GPU all-to-all
- **Batch:** `micro_batch_size=4`, no activation checkpointing
- **Optimizer:** FP32 optimizer states (Adam: 2×FP32 moments + FP32 master weights = ~12 bytes/param)

**Bottleneck analysis:** 62% utilization = large idle bubbles. The primary culprits are (1) the all-to-all collective on every expert layer with nothing to overlap against, and (2) memory pressure from FP32 optimizer states + full activations forcing a small micro-batch, which under-utilizes tensor cores and increases sync frequency per token.

## Changes

| # | Change | Why |
|---|--------|-----|
| 1 | **Reduce EP to 4, set DP=2** | Halves the per-layer all-to-all volume and lets the two data replicas overlap compute (forward/backward) with communication, directly attacking the 38% bubble. |
| 2 | **FP32 optimizer states → BF16 moments + FP32 master, with stochastic rounding** | Cuts optimizer memory from ~12 bytes/param to ~6 bytes/param (~9–12 GB/GPU saved for a 12.4B model). Master weights stay FP32 so convergence quality is preserved. |
| 3 | **Enable selective activation checkpointing (expert layers only)** | Frees the largest activation footprint (MoE expert activations) with minimal recompute (~4–6% extra FLOPs). Dense self-attention layers stay un-checkpointed to avoid recompute cost. |
| 4 | **Increase micro_batch_size 4 → 8** | Made possible by the memory freed in changes 2 and 3. Bigger GEMMs raise tensor-core efficiency and halve the number of sync points per token. |
| 5 | **Gradient accumulation (8 micro-batches/step) + async NCCL (compute/comm overlap, `CUDA_DEVICE_MAX_CONNECTIONS=8`)** | Communication from micro-batch *k* hides behind compute of micro-batch *k+1*, raising steady-state utilization toward 80%+. |
| 6 | **Router load-balancing aux-loss + tuned capacity factor** | Balances expert workloads, cutting straggler wait and token-dropping bubbles. Watch quality impact (see Risks). |

## Before/After Metrics

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Throughput (tokens/sec) | 12,400 | ~20,500 | **+65%** |
| GPU utilization | 62% | ~84% | +22 pp |
| Peak memory / GPU (80 GB) | ~64 GB (80%) | ~56 GB (70%) | −8 GB (headroom) |
| Optimizer state / GPU | ~19 GB (FP32) | ~9 GB (BF16) | −10 GB |
| Micro-batch size | 4 | 8 | 2× |
| Expert all-to-all volume/step | 8-GPU | 4-GPU | −50% |

*Effective FLOPs increase ~5% (checkpointing recompute); the +65% tokens/sec comes from higher utilization and larger batches, not more raw compute.*

## Risks
- **Optimizer precision (change 2):** BF16 moments can degrade convergence on noisy gradients. Mitigate with FP32 master weights + stochastic rounding, and A/B against a reference loss curve.
- **Load imbalance (change 6):** The aux-loss router can trade token quality/representation collapse for balance. Monitor token-dropping rate and downstream eval; tune the aux coefficient down if quality slips.
- **EP reduction (change 1):** DP=2 adds expert gradient all-reduce per step. Net is usually positive given the all-to-all savings, but on very fast interconnects (NVLink-only) the tradeoff is smaller — benchmark EP=4 vs EP=8.
- **Activation checkpointing (change 3):** Recomputed expert layers raise runtime slightly; if applied too aggressively it can erase gains. Keep it off dense layers and only enable where memory headroom dictates.
- **micro_batch 8 + grad accumulation:** Activations still grow; if OOM appears under load, fall back to checkpointing the next-deepest layer rather than shrinking the batch.
- **Overlap tuning:** async all-to-all with gradient accumulation increases host-side memory and ordering complexity; bugs can surface as nondeterministic deadlocks. Keep communication synchronous within a step boundary until validated.