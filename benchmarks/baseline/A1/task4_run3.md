I'll research MoE throughput optimization techniques to ground this report, then produce it.
Let me gather a bit more grounding on optimizer memory savings and activation checkpointing impact.
# MoE Training Throughput Optimization Report

**Target:** 8x GPU MoE training run
**Baseline:** 12,400 tokens/sec · 62% GPU util · `expert_parallel=8`, `micro_batch_size=4`, no activation checkpointing, FP32 optimizer states

## Changes

| # | Change | From → To | Rationale |
|---|--------|-----------|-----------|
| 1 | Rebalance parallel strategy | `EP=8, DP=1` → `EP=2, DP=4` | EP=8 on 8 GPUs means each GPU hosts very few experts (1 expert/GPU for an 8-expert model). Token-routing skew then maps **directly to GPU idle time**, and the all-to-all dispatches tokens across all 8 GPUs. Reducing EP to 2 (or 4) keeps several experts per GPU, which **statistically averages out routing imbalance**, shrinks the all-to-all domain (8-way → 2-way), and adds a data-parallel axis so every GPU does more real work per step. Hybrid EP+DP beat pure EP in published benchmarks (e.g. 8.77x vs 8.34x in a 4-GPU EP/DP study); this is the primary fix for the 62% utilization. |
| 2 | Enable activation checkpointing | off → selective/full recompute | Frees the activation buffer that currently caps the micro-batch, directly enabling Change 4. |
| 3 | Switch optimizer precision | FP32 states → mixed precision (BF16 compute, FP32 master weights, BF16/FP16 Adam moments via distributed optimizer) | Adam's FP32 `m`+`v` dominate model-state memory (~8 B/param). Halving optimizer states cuts ~25–40% of peak memory; on A100/H100-class GPUs BF16 GEMMs run 2–4x faster than FP32 on tensor cores. Keep FP32 master weights to protect convergence. |
| 4 | Increase micro-batch size | 4 → 8–16 (keep global batch constant via gradient accumulation) | MBS=4 yields small GEMMs and poor tensor-core / bandwidth efficiency. Larger micro-batches amortize kernel launch and all-to-all latency; larger expert GroupGEMMs (more tokens per expert) are precisely where MoE throughput wins. |
| 5 | Overlap communication with compute | sync all-to-all → async/overlapped dispatch (DeepEP-style, or Megatron comm overlap) | Hides all-to-all latency behind FFN compute; required to hold high utilization after Changes 1–4. |
| 6 | Enable load-balancing control | default router → `--moe-router-force-load-balancing`, tuned `aux_loss_coef`, adequate capacity factor | Prevents expert collapse and OOM during the unstable first ~100–300 training steps; with EP reduced this is what keeps routing skew from re-creating the imbalance. |

**Optional:** add `TP=2` (`EP=2, DP=2, TP=2`) if expert weights no longer fit per GPU after reducing EP, or to further shard large expert FFNs.

## Before / After Metrics

*Estimates derived from the stated baseline and published speedup ranges for EP→EP+DP rebalancing (~1.4–1.7x) and mixed-precision memory savings; validate on a 50–100 step A/B run before committing.*

| Metric | Before (baseline) | After (projected) | Delta |
|--------|-------------------|-------------------|-------|
| Throughput (tokens/sec) | 12,400 | ~18,600–21,100 | **+50–70%** |
| GPU utilization | 62% | ~88–92% | **+26–30 pts** |
| Peak GPU memory | High (FP32 opt states, full activations, MBS=4 limit) | ~20–40% lower for same batch; MBS up to 8–16 fits | **↓ memory, ↑ batch** |
| Expert load balance (per-GPU tokens) | High skew (1 expert/GPU, EP=8) | Near-uniform (4 experts/GPU, EP=2 + aux loss) | balanced |

## Risks

- **Load imbalance / expert collapse:** Reducing EP improves balance statistically but changes routing dynamics; if the router collapses, imbalance (and OOM) returns. Mitigate with `--moe-router-force-load-balancing`, `aux_loss_coef` tuning, and capacity-factor checks — especially for the first few hundred steps from a cold start.
- **Numeric drift from precision:** BF16/FP16 optimizer states cut mantissa bits; without FP32 master weights updates can vanish or diverge. Requires fp32 master copies, fp32 LayerNorm/softmax/loss accumulation, and (for FP16 only) loss scaling. Watch loss/val curves — convergence changes are the top silent regression risk.
- **Recompute overhead:** Activation checkpointing trades memory for ~20–40% extra compute; gains only materialize if Change 4 (larger MBS) actually increases utilization enough to offset it.
- **Parallel-strategy side effects:** Changing EP/DP alters per-GPU expert residency (more expert weights per GPU when EP drops — memory-up if the model was EP=8 to *fit* weights) and changes gradient statistics/batch composition. Keep effective global batch constant via gradient accumulation; reproducibility requires matched seeds/dataloader order.
- **Communication hazards:** Async/overlapped all-to-all needs framework support; misuse can deadlock or race. Lower all-to-all degree (EP=2) reduces traffic but also reduces the comm/compute overlap window, so pipelining must actually engage.
- **Hardware dependence:** BF16 requires Ampere+ (A100/H100). On older GPUs (V100/RTX) fall back to FP16 + loss scaling or skip Change 3.
- **Estimates are unvalidated:** All "After" figures are projections; the safe protocol is a short A/B benchmark with `nvidia-smi dmon` + NCCL profiling before the full run.