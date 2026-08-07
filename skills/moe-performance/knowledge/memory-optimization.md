# MoE memory optimization

MoE training memory has the usual dense-model components plus
capacity-factor-driven buffers and all-to-all buffers that a dense model never
sees. This note maps where the memory goes and how each knob in the
optimization catalog moves it. All formulas match the `moe-training` memory
estimator (`skills/moe-training/tools/memory_estimator.py`); run it per layout
change rather than trusting a hand count.

## Where MoE memory goes

| Component | MoE-specific behavior | Sharded by |
| --- | --- | --- |
| Parameters (bf16 = 2 B/param) | Expert parameters are **replicated** across data-parallel groups (every DP replica holds the full expert set unless EP shards it) and **sharded** by EP within a group: `expert_params / ep` | Dense: `dp × tp × pp`; experts: `ep` |
| Optimizer states | AdamW mixed precision ≈ 12 B/param (fp32 master + 2 moments); sharded the same way as parameters | Same as parameters |
| Gradients | Same footprint as parameters, per owned parameter | Same as parameters |
| Activations | Scale with `num_layers × micro_batch × seq_len × d_model / tp`; **not** reduced by expert count | Tensor parallelism |
| Capacity-factor buffers | Per-expert input/output buffers sized by `capacity_factor × tokens / num_experts`; extra capacity is padding in memory even when idle | EP group |
| All-to-all buffers | Dispatch/receive staging buffers sized by `top_k × tokens_per_step × dtype_bytes` | EP group |

The expert-redundancy term is the trap: with EP=1 (no expert parallelism),
every data-parallel rank stores all 6 B of expert weights, so total expert
memory is `expert_params × dp`, not `expert_params`. Raising EP removes that
redundancy.

## Activation memory

Activations are typically the largest *controllable* term. The estimator's
rough formula:

```text
activation_bytes_per_gpu = num_layers × micro_batch × seq_len × d_model × 20
                           / tp × (0.5 if recompute else 1.0)
```

The knobs that move it, in order of leverage:

- **Micro-batch size** — linear; halving it halves activation memory.
- **Sequence length** — linear; context is usually set by the task, so this is
  rarely free.
- **Gradient checkpointing (activation recomputation)** — halves activations
  by recomputing them in the backward pass. Cost: ~20–30% recompute FLOPs.
- **Tensor parallelism** — divides activations by `tp`; needs a fast
  intra-node interconnect.

Capacity factor and top-k do not show up in this line — the per-expert
activation buffers (next section) are where they bite.

## Capacity factor tuning

The capacity factor sets how much per-expert buffer is allocated:

```text
per_expert_buffer = capacity_factor × tokens_per_expert_balanced
```

There is a three-way trade: **memory / compute / quality**.

- **Lower capacity factor (1.25 → 1.0)** — shrinks every per-expert buffer and
  the all-to-all volume by up to 20%, and removes padding compute. Cost:
  if routing is imbalanced, overflow tokens are dropped (quality loss).
- **Higher capacity factor** — absorbs routing spikes without drops. Cost:
  memory, padding compute, and dispatch volume all grow.
- **Load balancing** is what makes lowering the capacity factor safe: a
  balanced router at capacity 1.0 drops almost nothing, so the memory/compute
  win is nearly free.

Sequence packing is the memory-side complement: packed sequences fill the
capacity exactly, so the capacity factor no longer has to inflate buffers to
absorb tail padding.

## Offload and sharding

When memory still does not fit:

- **Expert offload to CPU** — move the expert weights (and optionally their
  optimizer states) to host memory and stream them per MoE layer. Expert
  parameters are offloaded (recomputed) per forward/backward pass. Trade-off:
  fits memory, but adds host↔device traffic; only sensible when experts are
  rarely touched.
- **ZeRO offload** — offload optimizer states (stage 2) or parameters +
  gradients + optimizer states (stage 3) to CPU. Trade-off: fits memory,
  slower steps due to CPU↔GPU round trips; fine when memory, not step time,
  is the binding constraint.
- **EP sharding** — spread `num_experts` across ranks so each holds
  `num_experts / ep`. Trade-off: removes per-rank expert redundancy but adds
  an all-to-all per MoE layer (see `knowledge/communication.md`).

General rule: shard what you can (EP, ZeRO), offload what you cannot, and
measure the step-time cost of each — offload trades memory for latency, so it
is the last resort on a throughput-bound run.

## Practical guidance

Checklist for a memory-bound MoE run:

1. Run the `moe-training` memory estimator on the exact parallelism layout;
   confirm whether params/optimizer or activations dominate.
2. If activations dominate — enable gradient checkpointing first (cheapest
   halving), then shrink micro-batch, then raise `tp` if interconnect allows.
3. If expert redundancy dominates (large `dp`, small `ep`) — raise EP; this is
   usually a communication trade, not a free win.
4. If capacity buffers dominate — tighten the capacity factor toward 1.0, but
   only after load balancing (else dropped tokens); use sequence packing to
   remove padding.
5. Only if the above are exhausted — offload experts or optimizer states to
   CPU, accepting slower steps.
6. Cross-check the outcome with the `moe-debugging` OOM workflow: a run that
   still OOMs after these steps is a capacity-factor or micro-batch problem,
   not a sharding problem.
