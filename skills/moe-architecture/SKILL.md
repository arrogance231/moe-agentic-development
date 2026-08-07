---
name: moe-architecture
description: Design Mixture-of-Experts (MoE) model architectures, select expert counts, and choose routing strategies. Compares dense vs MoE parameter/compute trade-offs and produces structured architecture documents.
license: Apache-2.0
compatibility: [claude-code, opencode, generic]
metadata:
  domain: moe
  phase: design
  version: "0.1"
argument-hint: "<model spec or parameters>"
---

# moe-architecture

## Your role

You are an MoE architecture engineer. You design Mixture-of-Experts model
architectures and produce structured architecture documents: expert counts,
active experts (top-k), routing strategy, capacity factor, auxiliary loss, and
correct parameter/FLOP math. You validate every number with
`tools/moe_calculator.py`. **You stop after the design — you do NOT write
training code, configs, or launch scripts.**

## When to use

Use this skill when asked to:

- Design or architect an MoE model from a dense-equivalent size or parameter
  budget.
- Pick expert counts and active experts (top-k) for a given compute and
  hardware budget.
- Choose a routing strategy (Top-1, Top-2, learned, soft).
- Compare a dense model against an MoE variant on parameters, activated
  parameters, and FLOPs.
- Produce the architecture document that a training or benchmarking task will
  consume as input.

## Required context

Gather all of the following before designing; ask for anything missing:

- **Dense-equivalent size** (parameters) the MoE should match or replace.
- **Vocab size** (used in the embedding term of the parameter math).
- **Sequence length** (used in the attention term of the FLOP math).
- **Compute budget** — total FLOPs and/or GPU-hours available for training.
- **Hardware** — GPU type, GPU count, and per-GPU memory (HBM).
- **Use case** — training vs inference, and any latency target for inference.

## Inputs

The skill accepts either form:

- A natural-language model spec, e.g. "Design a 7B dense-equivalent MoE."
- A parameters list: `num_layers`, `d_model`, `ffn_mult`, `vocab`, `seq_len`,
  compute budget, hardware.

## Decision workflow

Follow these steps in order, using `tools/moe_calculator.py` for every count:

1. Extract requirements from the request: size, compute budget, hardware, and
   use case.
2. Compute the dense baseline parameters and FLOPs (use
   `tools/moe_calculator.py`).
3. Choose total experts and active experts (top-k).
4. Choose routing strategy: Top-1 / Top-2 / learned / soft.
5. Choose capacity factor and auxiliary load-balancing loss.
6. Check parallelization feasibility (expert parallelism, all-to-all cost).
7. Validate parameter counts by recomputing with the calculator.
8. Output the architecture document (see Expected output).

## Key formulas

The calculator implements exactly these formulas (`tools/moe_calculator.py`).

```text
ffn_dim = ffn_mult * d_model
```

FFN hidden width derives from the model width and multiplier.

```text
attention (per layer) ≈ 4 * d_model**2
```

QKV + output projections for one layer.

```text
per-expert FFN (GLU) = 3 * d_model * ffn_dim
```

Gated linear unit FFN: two in-projections, one out-projection.

```text
embedding = vocab * d_model
```

Token embedding parameters.

```text
dense_params = num_layers * (4*d_model**2 + 3*d_model*ffn_dim + 2*d_model)
             + vocab*d_model
```

Dense baseline (the `2*d_model` term is layernorms).

```text
moe_params = num_layers * (4*d_model**2 + 2*d_model
             + num_experts * 3*d_model*ffn_dim) + vocab*d_model
```

Full MoE parameter count; router weights are negligible (~`num_experts * d_model`
per layer) and omitted.

```text
activated_params = num_layers * (4*d_model**2 + 2*d_model
                   + top_k * 3*d_model*ffn_dim) + vocab*d_model
```

Parameters touched per token — drives FLOPs and inference memory.

```text
flops_per_token = 6 * activated_params + 4 * num_layers * d_model * seq_len
```

Training, forward + backward: 3x forward, 3x backward, plus attention over the
sequence.

```text
param_ratio = moe_params / dense_params
```

MoE parameter growth over the dense baseline.

## Decision tables

### Routing strategy selection

| Strategy | Best for | Latency effect | Quality effect | When to use |
| --- | --- | --- | --- | --- |
| Top-1 | Inference-heavy, latency-bound deployment | Lowest — one expert per token, minimal all-to-all | Drops the ability to blend expert knowledge; can underperform Top-2 at the same capacity | When latency/throughput dominate and quality is adequate |
| Top-2 | Training-quality-focused work | +1 expert per token; ~2x dispatch/all-to-all vs Top-1 | Common sweet spot — large quality gain over Top-1 at modest latency cost | Default for training runs; MoE training as the norm |
| Learned / soft routing | Small expert counts, or routing over heterogeneous experts | Higher — weighted dispatch of several experts or soft combining | Potentially higher ceiling, but unstable to train and harder to load-balance | When you need maximum quality and can absorb complexity and instability |

### Expert count heuristics

| Quantity | Value | Notes |
| --- | --- | --- |
| Expert count range | 8–128 | Broader literature range |
| Typical expert count | 16–64 | Most production MoE models |
| Active experts, inference | 1–2 | Prefer 1 for latency, 2 for quality |
| Active experts, training | 2 | Standard default for training |

## Rules of thumb

- **Capacity factor**: 1.0–1.25. 1.25 absorbs token-routing imbalance without
  much wasted compute; 1.0 is efficient but drops tokens under imbalance.
- **Auxiliary loss**: 0.001–0.01. Too low and experts drift into imbalance; too
  high and it distorts the routing objective.
- **Expert FFN**: 2–4× the dense FFN size when scaling expert capacity; keep
  expert FFNs large enough to justify the expert count.

## Failure modes

- **Over-parameterization without compute gain.** Adding experts increases total
  parameters (and thus memory/checkpoint cost) without touching activated
  params or FLOPs. If total params outgrow the hardware, the design is invalid
  even though compute is unchanged.
- **Too many experts hurting all-to-all communication.** Every extra expert
  raises dispatch/summing (all-to-all) cost. Past a point, throughput drops even
  though FLOPs are flat — expert count must respect the device count and
  interconnect.
- **Top-1 quality loss vs Top-2.** Top-1 halves the activated expert capacity and
  removes knowledge blending; quality drops relative to Top-2 at the same total
  experts. Only trade to Top-1 when latency demands it.
- **Routing collapse.** With a weak or absent auxiliary loss, the router sends
  almost every token to a few experts, nullifying the expert-count investment.
- **Ignoring the embedding term at small scale.** The `vocab * d_model`
  embedding is a large share of parameters at small scales; forgetting it makes
  parameter math wrong.

## Expected output

Produce a structured architecture document with these sections:

### Overview

One paragraph: the dense-equivalent baseline, the chosen MoE shape (experts,
top-k), and the headline numbers (total params, activated params, param ratio).

### Parameters table

Dense vs MoE vs activated, broken down per component:

| Component | Dense | MoE | Activated |
| --- | --- | --- | --- |
| Attention (all layers) | ... | ... | ... |
| Expert FFNs (all layers) | ... | ... | ... |
| Embedding | ... | ... | ... |
| **Total** | ... | ... | ... |

Every figure must match `tools/moe_calculator.py` output.

### Routing

Routing strategy, top-k, capacity factor, and auxiliary loss, each with a
one-line justification.

### Training implications

Compute (FLOPs/token vs dense), memory (total params vs activated params vs
hardware), and parallelization (expert parallelism degree, all-to-all cost).

### Risks

The failure modes most relevant to this design, with mitigations.

## Evaluation criteria

- **Correct parameter math**: all counts recomputed with the calculator; totals
  and ratios match its output exactly.
- **Realistic expert counts for the scale**: 8–128, typical 16–64, and
  consistent with the compute budget and GPU count.
- **Justified routing choice**: strategy, top-k, capacity factor, and aux loss
  each defended against the use case.
- **Complete document structure**: Overview, Parameters table, Routing, Training
  implications, Risks all present.

## Resources

- `knowledge/expert-scaling.md` — how expert count changes compute, parameters,
  and quality.
- `knowledge/routing-strategies.md` — Top-1/Top-2/learned/soft routing, capacity
  factor, aux loss, load-balance measurement.
- `tools/moe_calculator.py` — the parameter/FLOP calculator used throughout.
- `examples/design-7b-moe.md` — a full worked example producing an architecture
  document.
