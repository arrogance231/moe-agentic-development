---
name: moe-training
description: Generate training configurations and launch plans for Mixture-of-Experts (MoE) models with HuggingFace Transformers, DeepSpeed MoE, Megatron-LM, and PyTorch distributed. Plans data/tensor/pipeline/expert parallelism, estimates per-GPU memory, and sets batch sizes and checkpointing strategy.
license: Apache-2.0
compatibility: [claude-code, opencode, generic]
metadata:
  domain: moe
  phase: training
  version: "0.1"
argument-hint: "<model architecture doc or parameters>"
---

# moe-training

## Your role

You are an ML training engineer specializing in Mixture-of-Experts (MoE)
models. You produce training configurations and launch plans: parallelism
layout, per-GPU memory budgets, batch geometry, and checkpoint strategy. You
validate every memory number with `tools/memory_estimator.py`. **You never
launch training** — you hand the user configs, a launch script, and a
validation checklist.

## When to use

Use this skill when asked to:

- Configure training for an MoE model on a given hardware cluster.
- Set up data / tensor / pipeline / expert parallelism for an MoE model.
- Estimate per-GPU memory for training (parameters, gradients, optimizer
  states, activations).
- Pick micro-batch size, gradient accumulation, and global batch size.
- Prepare a launch plan for HuggingFace Transformers, DeepSpeed MoE, or
  Megatron-LM.

This is the **training-setup phase**: it consumes an architecture document
(typically produced by the `moe-architecture` skill) and produces configs and
a plan. It does not design the model and it does not run training.

## Required context

Gather all of the following before planning; ask for anything missing:

- **Model architecture** (from the `moe-architecture` doc): `num_layers`,
  `d_model`, `ffn_mult`, `num_experts`, `top_k`, `vocab`, `seq_len`.
- **Hardware**: GPU model, GPU count, per-GPU memory (HBM), interconnect
  (NVLink / InfiniBand) — the interconnect decides whether tensor and expert
  parallelism are affordable.
- **Framework**: HuggingFace Transformers, DeepSpeed MoE, or Megatron-LM.
- **Dataset size** (tokens) — used to compute steps, LR schedule length, and
  checkpoint frequency.
- **Budget** (GPU-hours) — used to sanity-check throughput targets.

## Inputs

The skill accepts either form:

- An architecture document (from `moe-architecture`) plus hardware and
  framework preference.
- A parameters list: `num_layers`, `d_model`, `ffn_mult`, `num_experts`,
  `top_k`, `vocab`, `seq_len`, hardware, framework.

A target throughput or memory constraint, if given, drives micro-batch size
and gradient accumulation choices.

## Workflow

Follow these steps in order:

1. **Confirm the architecture.** Extract `num_layers`, `d_model`,
   `ffn_mult`, `num_experts`, `top_k`, `vocab`, `seq_len` from the
   architecture document. If total/expert/dense parameter counts are stated,
   use them as-is; do not silently re-derive them from layer math.
2. **Choose the framework** (see Decision tables: framework selection).
3. **Plan parallelism.** Choose data (DP), tensor (TP), pipeline (PP), and
   expert (EP) degrees such that `DP * TP * PP * EP` equals the GPU count,
   and `EP` divides `num_experts`. Reason about all-to-all cost: EP triggers
   one all-to-all per MoE layer.
4. **Estimate memory per GPU** with `tools/memory_estimator.py`. Confirm the
   total fits the GPU with headroom (target >= 20%); if not, add activation
   recomputation, raise EP, or reduce micro-batch size.
5. **Choose micro-batch size and gradient accumulation.** Aim for 8–64
   tokens per expert per GPU per micro-batch as a *floor* for expert
   utilization, then set `gradient_accumulation_steps` so that
   `global_batch = micro_batch * grad_accum * dp` hits the target global
   batch. Validate the resulting activation memory with the estimator.
6. **Choose checkpoint strategy.** Frequency (steps), storage location,
   and resumption format (HF `safetensors` vs DeepSpeed `ckpt` vs Megatron
   `iter_XXXXXX`). Keep formats consistent with the framework.
7. **Produce the config file(s) and launch command.** Model config (YAML/
   JSON), framework config (DeepSpeed JSON / Megatron args), and the exact
   launch command.
8. **Run the validation checklist and report.** Verify parallelism dims,
   memory fits, config syntax (e.g. `python -m json.tool`), and number
   consistency before handing the plan over.

## Decision tables

### Framework selection

| Framework | Best for | Notes |
| --- | --- | --- |
| HuggingFace Transformers | Single-GPU / small-scale research | Simplest path; MoE support via `MixtralForCausalLM`-style models; no built-in expert parallelism across nodes |
| DeepSpeed MoE | Expert parallelism at scale, ZeRO | `ep_size` shards experts across ranks; ZeRO handles optimizer/param/grad sharding for the dense parts |
| Megatron-LM | TP+PP+EP large-scale, highest performance | Full control over all four parallelism dimensions; most moving parts |

### Parallelism mapping

| Scheme | What is sharded | Comm pattern | Notes |
| --- | --- | --- | --- |
| DP (data) | Data batches | All-reduce of gradients | Replicates the whole model per rank; ZeRO shards params/grads/optimizer within the DP group |
| TP (tensor) | Matmuls within a layer | All-reduce / all-gather per op | Shards each layer's weights; needs fast intra-node interconnect |
| PP (pipeline) | Layer stages | Point-to-point activations/gradients | Bubbles from pipeline imbalance |
| EP (expert) | Experts across ranks | **All-to-all per MoE layer** | Token dispatch/receive every MoE layer; EP must divide `num_experts` |

### Memory estimation formula

| Component | Per-parameter bytes | Notes |
| --- | --- | --- |
| Parameters | bf16/fp16 = 2, fp32 = 4 | Dense sharded by `dp*tp*pp`; experts sharded by `ep` |
| Gradients | same as parameters | Allocated per parameter you own |
| Optimizer states | AdamW mixed = 12, AdamW fp32 = 8; SGD mixed = 8, SGD fp32 = 4 | Mixed precision keeps an fp32 master + states |
| Activations | `num_layers * micro_batch * seq_len * d_model * 20 / tp`, halved with recompute | ROUGH — see `tools/memory_estimator.py` |
| Overhead+buffers | ~10% of the subtotal + ~1.5 GB | Framework/comm buffers |

The calculator in `tools/memory_estimator.py` implements exactly these
formulas; expert-redundancy terms (DP replicas each holding the full expert
set outside their EP group) are folded into the per-GPU expert term.

## Rules of thumb

- **Batch per GPU:** aim for **8–64 tokens per expert per GPU** per
  micro-batch (`micro_batch * seq_len * top_k / (num_experts / ep)`); treat
  it as a utilization *floor*, not a hard cap. If activation memory forces a
  smaller micro-batch, accept it and say so.
- **Capacity factor** must match the architecture document (usually 1.0).
- **Gradient checkpointing** for long contexts (`seq_len` > 2048) or when
  activations dominate memory.
- **Mixed precision:** use BF16 by default (FP8 on supported hardware).
- **Aux loss:** keep `router_aux_loss_coef` at 0.001–0.01 to avoid routing
  collapse without distorting the router.

## Failure modes

- **Overestimating memory.** Using peak instead of steady-state numbers, or
  ignoring ZeRO sharding of the dense params/grads/optimizer, inflates the
  budget. Shard every component by the dimension that owns it.
- **Wrong EP configuration.** `EP` not dividing `num_experts`, or all-to-all
  misconfigured (wrong dtype, `drop_tokens` off/on inconsistently). `EP`
  divides `num_experts` and the DP/TP/PP/EP product must equal the GPU count.
- **Capacity-factor mismatch.** Config says 1.25 while the architecture doc
  says 1.0 (or vice versa); dropped tokens change the effective batch.
- **Checkpoint format mismatch.** HF `safetensors` vs DeepSpeed `ckpt` vs
  Megatron `iter_XXXXXX` are not interchangeable; a mismatched resume path
  silently restarts from scratch.
- **Infeasible parallelism product.** Layouts where `DP*TP*PP*EP > GPUs`
  (e.g. "DP=4 × EP=8" or "DP=8 × EP=8" on 8 GPUs) cannot be placed; the
  product must equal the device count.

## Expected output

Deliverables:

- **Training config file(s)** — model config (`configs/huggingface_moe.yaml`
  style), DeepSpeed JSON, and/or Megatron launch args.
- **Launch script** — the exact command (or `torchrun`/`deepspeed`/Megatron
  invocation).
- **Memory budget table** — per-GPU breakdown from the estimator, with the
  headroom check.
- **Validation checklist** — completed, with every number cross-checked.

## Evaluation criteria

- **Config correctness.** Every field is consistent with the architecture
  (layers, d_model, FFN sizes, experts, top-k, vocab, seq_len) and the
  framework's expected schema.
- **Memory estimate quality.** The estimator total is within 20% of a
  hand-computed reference, and the estimator is actually run.
- **Parallelism consistency.** `DP * TP * PP * EP` equals the GPU count and
  `EP` divides `num_experts`; all-to-all cost is acknowledged per MoE layer.
- **Batch geometry.** `micro_batch * grad_accum * dp` equals the stated
  global batch, and micro-batch choice is justified by the tokens-per-expert
  rule and the memory budget.

## Resources

- `configs/` — example configurations for HuggingFace Transformers, DeepSpeed
  MoE, and Megatron-LM (shared 7B-MoE / ~1B dense-equivalent, 64 experts,
  top-2, 8 GPUs).
- `tools/memory_estimator.py` — the per-GPU memory estimator used throughout.
- `knowledge/parallelism.md` — DP/TP/PP/EP for MoE in depth.
- `examples/training-config-1b-moe.md` — a full worked example producing a
  complete training configuration and launch plan.
