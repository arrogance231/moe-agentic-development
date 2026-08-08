# MoE Throughput Optimization Report

Scenario: 12,400 tokens/sec, 8 GPUs, 62% GPU util, `expert_parallel=8`, `micro_batch_size=4`, no activation checkpointing, FP32 (mixed) AdamW optimizer states. Working assumption for memory math: 64 experts, top-2, ~1.9B total params, `d_model=2048`, 16 layers, `seq_len=2048`. All figures are estimates pending confirmation with `profilers/throughput_profiler.py` and `tools/memory_estimator.py`.

**Diagnosis:** Memory is not binding (~16.5 GB/GPU ≈ 21% of an 80 GB HBM). The 62% util at EP=8 with `micro_batch_size=4` points to a communication-bound all-to-all (EP=8 spreads every dispatch across all ranks) plus low matmul efficiency from a small micro-batch and padding from expert skew.

## Changes

1. **Enable activation checkpointing** (`gradient_checkpointing: true`). Activations are the only headroom available; checkpointing halves peak activation memory (≈5.4 → 2.7 GB at the current batch), funding the batch increase below at flat memory. Adds +20–30% recompute, but the run is comm/util-bound, not compute-bound.
2. **Raise `micro_batch_size: 4 → 8`** (halve `gradient_accumulation_steps` to keep global batch constant). Larger batches give better matmul shapes and more tokens per expert per GPU, lifting utilization directly. With checkpointing, activation memory lands back at ≈5.4 GB — the checkpoint gain is fully re-spent on batch, memory stays flat.
3. **Lower `capacity_factor: 1.25 → 1.0`** once load is balanced. Cuts dispatch/all-to-all volume and padding waste; +3–8% tokens/sec. Do this *after* #4 so imbalance doesn't drop tokens.
4. **Raise `router_aux_loss_coef: 0.001 → 0.01`.** Reduces expert skew → less capacity padding, higher effective expert utilization (+5–10%).
5. **Sequence packing** (or dynamic capacity). Pack to capacity instead of padding to the capacity factor; +10–15% tokens/sec. Highest-value batching lever for this run.
6. **Grouped matmuls for expert FFN** (implementation-level). Batch all experts' FFN inputs into grouped GEMMs instead of per-expert loops; +10–30% on the expert-FFN region (kernel/framework dependent).
7. **Optional A/B — `expert_parallel 8 → 4`** (with `DP=2`). Halves all-to-all volume per step at the cost of 2× experts per rank (≈+3 GB/GPU); only adopt if step-time A/B shows communication is dominant.

## Before/After Metrics

| Metric | Before | After (plan) | Delta |
| --- | --- | --- | --- |
| Tokens/sec (global) | 12,400 | ~16,800 | **+35%** |
| GPU utilization | 62% | ~78% | +16 pts |
| Bubble time (idle %) | ~38% | ~22% | −16 pts |
| Peak activation mem/GPU | ~5.4 GB | ~5.4 GB (checkpointed, mb=8) | flat |
| Total est. mem/GPU | ~16.5 GB | ~16.5–17.5 GB | flat (not binding) |
| Effective experts | below n (skew) | ≥ 0.5·n | quality/balance |

Estimated contributions (conservative, non-additive in practice): micro-batch 4→8 ~+15%, sequence packing ~+10%, capacity-factor + aux-loss ~+8%, grouped matmuls ~+5–8%. Every number must be verified by re-running `throughput_profiler.py` before/after.

## Risks

- **Capacity factor 1.25 → 1.0** can drop tokens under residual imbalance and distort training — gate this change on measured load balance (effective experts ≥ 0.5·n) after the aux-loss bump.
- **Aux loss 0.001 → 0.01** can over-regularize the router and hurt routing quality if raised before the capacity factor drops.
- **Checkpointing adds +20–30% compute**; if the batch growth doesn't fully compensate, tokens/sec could plateau — verify with an A/B step-time run before committing.
- **micro_batch 8** could OOM if the activation estimate is off; keep the headroom check (target ≥ 20%) in the estimator before enabling.
- **Sequence packing** introduces attention-masking complexity and can distort loss weighting if not masked correctly.
- **EP 8 → 4** adds ~3 GB/GPU and more all-to-all recipients if misconfigured (`EP` must divide `num_experts`; `DP×TP×PP×EP` must equal 8) — treat as A/B only.
- **Grouped-matmuls** gains are kernel/framework-dependent; benchmark before claiming them.

Verification plan: capture baseline with `throughput_profiler.py` (tokens/sec, util, step time), then re-run after changes 1–2, after 3–4, and after 5–6, isolating each gain.