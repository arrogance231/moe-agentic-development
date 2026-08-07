# Worked example: designing a 7B dense-equivalent MoE

## Input

> "Design a 7B dense-equivalent MoE."

## Requirements extracted

- **Dense-equivalent size**: 7B parameters.
- **Vocab size**: 32,000.
- **Sequence length**: 2048.
- **Compute/hardware**: training on 8×H100-class GPUs (80 GB HBM each, ~640 GB
  total).
- **Use case**: general language modeling, training-focused (not a strict
  latency target).

## Step-by-step walkthrough

### Step 1 — Extract requirements

A 7B dense-equivalent model, trained for general language modeling on
8×H100-class GPUs. No hard inference latency target, so quality can be favored
over Top-1 dispatch savings.

### Step 2 — Dense baseline

Run `tools/moe_calculator.py` to find a dense baseline near 7B. A 32-layer /
`d_model` 4096 / `ffn_mult` 4 shape computes to **8,721,268,736** (8.7B) — above
target. Dropping to 26 layers of the same width lands at **7,110,606,848**
(~7.1B), which is the dense baseline for the design.

### Step 3 — Total experts and active experts

Start from the 8–128 range. A 64-expert variant of this shape computes to
**336,883,564,544** params (a 47.38x ratio). That fails the feasibility check in
step 6 even on weights alone: ~674 GB of bf16 weights exceed the cluster's
~640 GB, and the all-to-all cost of dispatching across 64 experts on 8 GPUs is
severe. This is the "over-parameterization without compute gain" failure mode.

Settle on **8 experts with top-k 2** (Mixtral-style): 43.75B total params
(6.15x the dense baseline), 12.35B activated, a ratio and footprint that fit
the hardware and the compute envelope.

### Step 4 — Routing strategy

**Top-2 learned routing** — the standard training default. The second expert
meaningfully improves quality over Top-1, and there is no latency constraint to
justify dropping to Top-1.

### Step 5 — Capacity factor and auxiliary loss

- **Capacity factor 1.25**: absorbs routing imbalance, keeps token dropping
  rare during training.
- **Auxiliary loss 0.01** (top end of the 0.001–0.01 range): with only 8
  experts, strong load balancing is cheap insurance against expert collapse;
  the load-balancing term also keeps the capacity factor effective.

### Step 6 — Parallelization feasibility

8 experts maps cleanly onto 8 GPUs with **expert parallelism EP=8** — one
expert replica per GPU per layer. Each token is dispatched to 2 experts
(top-k), so all-to-all traffic is 2 sends per token per MoE layer; with
EP=8 this is a full 8-way exchange and is the dominant communication cost.
Attention runs tensor-parallel across the 8 GPUs, and the optimizer state
(~700 GB for Adam at 43.75B params) must be sharded with ZeRO-3/FSDP.

### Step 7 — Validate parameter counts

Recompute the final shape with the calculator (see the Calculator run
section). Total (43,752,046,592), activated (12,345,098,240), and ratio
(6.15x) all match the design targets.

### Step 8 — Output the architecture document

See the Final architecture document section below.

## Calculator run

```text
$ python3 skills/moe-architecture/tools/moe_calculator.py \
    --num-layers 26 --d-model 4096 --ffn-mult 4 \
    --num-experts 8 --top-k 2 --vocab 32000 --seq-len 2048
```

Output:

```text
Estimated MoE architecture
==========================
  num_layers  : 26
  d_model     : 4096
  ffn_mult    : 4 (ffn_dim = 16,384)
  num_experts : 8
  top_k       : 2
  vocab       : 32000
  seq_len     : 2048

Parameter table
+-----------------------------+------------------+
| Metric                      |            Value |
+-----------------------------+------------------+
| Dense params                |    7,110,606,848 |
| MoE params                  |   43,752,046,592 |
| Activated params            |   12,345,098,240 |
| Param ratio (MoE / dense)   |            6.15x |
| FLOPs per token             |   74,943,004,672 |
| FLOPs per token (TFLOPs)    |              0.1 |
+-----------------------------+------------------+
```

**Sanity check of the ratio.** The spec-band 2–4× ratio is only reachable with
very small expert blocks or a handful of experts; at this scale the ratio
tracks the expert count because each full-size expert FFN
(`3 * d_model * ffn_dim`) dominates the per-layer attention block. 6.15x for
8 experts is the same class of ratio as Mixtral 8x7B (~47B total for a ~7B
dense-equivalent), i.e. a sane, hardware-feasible result. Activated params
(12,345,098,240) are 1.74x the dense baseline (7,110,606,848), so training
FLOPs scale by roughly that factor: **74,943,004,672** FLOPs/token vs
~43.5 GFLOPs/token for the 7.1B dense model — 1.75x the compute for 6.15x the
parameters.

## Final architecture document

### Overview

A 7B dense-equivalent MoE for general language modeling: 26 layers, `d_model`
4096, FFN hidden 16,384, with 8 experts per layer and top-2 routing. The model
holds **43.75B total parameters**, activates **12.35B per token**, and costs
**74.9 GFLOPs/token** in training — ~1.75x the compute of the 7.1B dense
baseline for 6.15x the capacity. It targets training on 8×H100-class GPUs.

### Parameters table

| Component | Dense | MoE | Activated |
| --- | --- | --- | --- |
| Attention, all layers | 1,744,830,464 | 1,744,830,464 | 1,744,830,464 |
| Expert FFNs, all layers | 5,234,491,392 | 41,875,931,136 | 10,468,982,784 |
| Layernorms, all layers | 212,992 | 212,992 | 212,992 |
| Embedding (32,000 × 4,096) | 131,072,000 | 131,072,000 | 131,072,000 |
| **Total** | **7,110,606,848** | **43,752,046,592** | **12,345,098,240** |

Router weights (8 × 4,096 per layer, ~0.85M) are negligible and omitted, per
the calculator.

### Routing

- **Strategy**: Top-2 learned routing — default for training; the second
  expert gives a consistent quality gain over Top-1 at modest dispatch cost.
- **top-k**: 2.
- **Capacity factor**: 1.25 — absorbs imbalance; expected to be tightened
  toward 1.0 as the auxiliary loss keeps load balanced.
- **Auxiliary loss**: 0.01 — the load-balancing scale (standard form:
  `aux_scale * num_experts * sum_e (f_e * P_e)`), held at the top of the
  0.001–0.01 range given the small expert count.

### Training implications

- **Compute**: 74,943,004,672 FLOPs/token (0.1 TFLOPs/token), 1.75x the dense
  baseline's ~43.5 GFLOPs/token. Activated params, not total params, drive the
  compute line.
- **Memory**: 43.75B params ≈ 88 GB in bf16 weights — fits across the 8×80 GB
  cluster. Adam optimizer state (~16 B/param ≈ 700 GB) must be sharded
  (ZeRO-3/FSDP) across the 8 GPUs. Activations scale with the 12.35B activated
  budget, independent of expert count.
- **Parallelization**: expert parallelism EP=8 (one expert per GPU per layer);
  tensor parallelism for attention; ZeRO-3 for the optimizer. All-to-all cost
  is the main overhead — 2 dispatches per token per MoE layer over an 8-way
  exchange.

### Risks

- **All-to-all contention**: EP=8 with top-2 means every token crosses the
  interconnect twice per layer; on a slow interconnect this, not FLOPs,
  becomes the throughput bottleneck. Mitigation: profile dispatch bandwidth,
  consider grouped (e.g. 2×4) expert layouts.
- **Expert collapse / imbalance**: with only 8 experts, a weak router can
  collapse load to 2–3 experts. Mitigation: the 0.01 aux loss plus monitoring
  the effective number of experts via the moe-debugging analyzers.
- **Capacity-factor waste**: 1.25 means up to 25% idle expert compute; if
  utilization looks healthy, tightening toward 1.0 recovers throughput.
- **Aux-loss distortion**: if 0.01 proves too strong, routing quality drops;
  tune in the 0.001–0.01 range rather than assuming a fixed value.
