# Expert scaling

How the expert count changes compute, parameters, and quality for a
Mixture-of-Experts model.

## Overview

Adding experts changes three things, and the three change at very different
rates:

- **Total parameters** scale with `num_experts`.
- **Activated parameters** scale with `top_k`, which is independent of
  `num_experts`.
- **Compute per token** (FLOPs) is roughly constant in expert count at fixed
  `top_k`.

| Quantity | Depends on | Changes when you add experts? |
| --- | --- | --- |
| Total params | `num_experts` | Yes — grows linearly |
| Activated params | `top_k` | No (as long as `top_k` is fixed) |
| FLOPs per token | `top_k` (+ attention) | No |
| Memory (weights, checkpoints) | `num_experts` | Yes — grows linearly |
| All-to-all / dispatch cost | `num_experts` | Yes — grows with dispatch width |

The design tension is that the thing you want more of (quality via capacity)
comes coupled to the thing you do not want (memory and communication), while
compute stays flat. `tools/moe_calculator.py` captures the parameter and FLOP
side of this exactly: run it at `--num-experts 8`, then at `--num-experts 64`,
and observe that MoE params climb while Activated params and FLOPs per token do
not move.

## Effect on parameters

Total parameters scale with `num_experts`:

```text
moe_params = num_layers * (4*d_model**2 + 2*d_model
             + num_experts * 3*d_model*ffn_dim) + vocab*d_model
```

Each expert carries a full FFN (`3 * d_model * ffn_dim`), so ten extra experts
means ten extra FFNs per layer. The shared attention and norm blocks
(`4*d_model**2 + 2*d_model`) and the embedding (`vocab * d_model`) are paid
once and do not scale with experts.

Activated parameters scale with `top_k` instead:

```text
activated_params = num_layers * (4*d_model**2 + 2*d_model
                   + top_k * 3*d_model*ffn_dim) + vocab*d_model
```

With `top_k = 2`, only two expert FFNs per layer contribute per token no matter
whether there are 8 or 128 experts in total. This is why MoE parameter ratios
(`moe_params / dense_params`) in practice run from roughly 2–6× at small expert
counts to 10–20×+ for "many-expert" designs such as DeepSeek-V3 (671B total vs
~37B active), while activated params stay at single-digit to low-double-digit
billions.

A practical consequence: at fixed `top_k`, going from 8 to 64 experts
quadruples the parameter budget — which quadruples optimizer state, activation
checkpoint storage, and the size of every checkpoint save — for zero compute
increase. The calculator makes this concrete: for a fixed shape, MoE params
move with expert count while activated params and FLOPs are flat.

## Effect on compute

FLOPs per token is driven by activated parameters, not total parameters:

```text
flops_per_token = 6 * activated_params + 4 * num_layers * d_model * seq_len
```

At fixed `top_k`, this is essentially constant in expert count. Doubling
experts from 16 to 32 does not double FLOPs; it doubles weights on disk and in
GPU memory, and it raises the all-to-all/dispatch bandwidth because tokens must
be gathered and scattered across more expert replicas.

The compute-relevant memory effects to watch:

- **Weights**: total params × bytes per param (bf16 = 2 B, fp32 = 4 B). This is
  the number that determines whether the model fits on the GPU cluster at all.
- **Optimizer state** (training): typically 12–16 bytes per parameter for Adam.
  Optimizer state alone can exceed the weight memory.
- **Activations**: dominated by `top_k`-activated experts, roughly independent
  of expert count.
- **Communication**: expert-parallel dispatch, all-to-all, grows with the
  product of expert count and tokens per step.

So the compute story is: FLOPs flat, memory/communication up. Use
`tools/moe_calculator.py` for the FLOPs figure and treat the memory line as
params × bytes.

## Effect on quality and scaling laws

Scaling-law reasoning explains why MoE works. Model quality tracks a
capacity-compute trade-off, and MoE breaks the dense coupling between
parameters and compute: quality gains come from capacity (parameter count)
**without** proportional compute.

The classic empirical observation from the MoE literature:

- **At matched compute**, sparse (MoE) models outperform dense models of the
  same training FLOPs — the extra parameters give more capacity per unit of
  compute.
- **At matched parameter count**, sparse models underperform dense models — the
  dense model spent its compute budget on the parameters that matter, while the
  sparse model under-utilizes its capacity.

The practical interpretation for a design:

- A well-sized MoE behaves like a dense model several times larger, for a small
  multiple of the compute. Mixtral 8x7B (8 experts, top-2, ~47B params) is the
  canonical example: ~13B activated params, and it performs in the class of
  much larger dense models.
- **Diminishing returns**: each additional expert adds a fixed amount of
  capacity but a growing communication and memory burden. The marginal quality
  per expert shrinks as the expert count grows.
- **Compute-bound vs param-bound regimes**: if the project is compute-bound
  (FLOPs are the scarce resource), MoE is attractive because it spends the FLOP
  budget on high-capacity activated params. If the project is param/memory-bound
  (weights or memory are the scarce resource), MoE is a bad fit — the dense
  model spends the same memory more effectively.
- **Sparse upcycling**: a trained dense model can be split into an MoE with
  little further training, preserving most of the dense model's quality — the
  cheapest way to get an MoE with reasonable quality.

Practical expert-count ranges from the literature: 8–128 experts, with 16–64
the common operating range for production models. Small models and
latency-critical inference skew low (8–16); very large training runs with
aggressive parallelism skew high (128+).

## Practical guidance

Pick the expert count from the compute budget, GPU count, and all-to-all
constraints, in this order:

1. **Fix the dense-equivalent size.** The dense baseline sets the compute
   envelope. Compute it with `tools/moe_calculator.py` first.
2. **Choose `top_k`.** Default 2 for training, 1 for latency-bound inference.
   This fixes activated params and thus FLOPs/token.
3. **Size the cluster budget.** Total params must fit: weights + optimizer +
   activations across the GPU count. `moe_params × (2 B weights + ~14 B Adam)`
   is a fast first check. This is usually the binding constraint on expert
   count.
4. **Respect expert parallelism.** Expert-parallel degree is bounded by the
   number of GPUs. With `G` GPUs, an expert count that is a small multiple of
   `G` balances load well; going far above it increases all-to-all traffic per
   expert replica.
5. **Scan the expert count.** Run `tools/moe_calculator.py` for candidate
   counts (e.g. 8, 16, 32, 64). Pick the largest count whose total params fit
   and whose all-to-all cost is acceptable — then drop one step if the comm
   budget is tight.

The calculator does not model communication; treat its MoE-params output as the
hard input to the memory check and pick the count that keeps params inside the
hardware while `activated_params` stays at the target compute level.
