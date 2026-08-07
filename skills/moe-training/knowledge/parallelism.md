# Parallelism for MoE training: DP, TP, PP, EP

## Overview

Dense models shard parameters and data across two or three axes (data,
tensor, pipeline). A Mixture-of-Experts model adds a fourth axis: **expert
parallelism (EP)**, which shards the expert FFNs across ranks. This changes
the game in two ways:

1. **The dense/experts split.** In a model like the reference 7B-MoE (~1B
   dense-equivalent), only ~1B parameters behave like a dense model. The
   other ~6B live in 64 experts per layer. Where the dense block can be
   replicated (DP) or sharded (TP/PP), the expert block is usually *sharded
   across ranks with EP* and never replicated unless the expert count is
   tiny.

2. **Communication.** Every MoE layer does token routing: a **dispatch
   all-to-all** sends each token to the ranks holding its top-k experts, and
   a **combine all-to-all** returns the expert outputs. That is two
   all-to-all collectives per MoE layer per micro-batch — the dominant
   communication cost of MoE training, and it does not exist in dense
   models.

The practical consequence: choosing a layout is a joint decision between
"how do the 4 dims multiply to the GPU count" and "how expensive is the
per-layer all-to-all at this EP degree and interconnect".

The reference layout used throughout this skill is `DP=1, TP=1, PP=1, EP=8`
on 8 GPUs: 64 experts over 8 ranks puts **8 experts per rank**, EP occupies
all 8 GPUs, and the all-to-all stays inside one node.

## Data parallelism (DP)

- The model (or, with ZeRO, parts of it) is **replicated** across the DP
  group; each rank sees a **different slice of the batch**.
- Gradients are **all-reduced** across the DP group each step.
- For MoE, DP interacts with EP: if DP > 1, each DP replica still must
  *have* the expert set or be inside an EP group. Outside an EP group, a DP
  replica holds the full expert set (expensive); inside an EP group, the
  experts are sharded and the tokens must be dispatched via all-to-all.
- **ZeRO stages** shard the optimizer state (stage 1), gradients (stage 2),
  and parameters (stage 3) across the DP group, trading communication for
  memory. This matters because with `DP=1, EP=8` the dense replica (1B
  params) is small, but a large dense baseline would quickly blow up the
  optimizer memory.

## Tensor parallelism (TP)

- **Matmuls within a layer** are sharded: the attention QKV/output and FFN
  weight matrices are split across TP ranks; each rank computes a partial
  result and an all-reduce combines it.
- TP requires **fast intra-node interconnect** (NVLink); it is rarely the
  right choice across slow links.
- For MoE, TP interacts with EP: you can TP the *shared* (dense) layers and
  then EP the *expert* FFNs. Megatron supports combining TP and EP, but the
  expert all-to-all operates on TP-sharded tensors, which complicates the
  communication. The reference layout uses `TP=1` to keep the all-to-all
  simple.

## Pipeline parallelism (PP)

- **Layer stages** are sharded: rank 0 owns layers 0–5, rank 1 owns 6–11,
  and so on; activations and gradients flow point-to-point between stages.
- The cost is **pipeline bubbles** — idle time while a micro-batch drains
  through the pipeline — plus one set of activations kept in flight per
  stage boundary (balanced by fewer layers' weights per rank).
- PP is orthogonal to EP: each pipeline stage can host its share of MoE
  layers, and EP sharding happens *within* a stage across the EP group.
  With only 24 layers and 8 GPUs, PP is unnecessary (each rank would own 3
  layers); the reference layout uses `PP=1`.

## Expert parallelism (EP)

- **Experts are sharded across ranks.** Each rank owns `num_experts / ep`
  experts per layer (8 experts per rank at 64 experts / EP=8).
- **All-to-all communication per MoE layer.** In the forward pass the
  router assigns each token to its top-k experts; a dispatch all-to-all
  moves each token to the rank(s) holding its experts, the experts compute,
  and a combine all-to-all returns the weighted outputs. Both happen **every
  MoE layer**, so the volume scales with `layers * tokens * top_k`, not with
  the number of all-to-all calls you configure.
- **Trade-off: communication volume vs activation memory.** Bigger EP (more
  ranks per expert group) means fewer experts per rank and smaller expert
  weight/activation footprint, but the all-to-all spans more ranks and the
  per-message token volume per rank can grow. Smaller EP keeps the all-to-all
  local but leaves every rank holding more expert parameters.
- **`EP` must divide `num_experts`** (64 % 8 == 0). If it does not, expert
  sharding is impossible without dropping experts.
- **Communication dtype.** Dispatch/combine all-to-all typically runs in
  **fp32** (`communication_data_type: "fp32"`) to keep router sums
  numerically safe, even when parameters are bf16.

## Combined layouts

Pick a layout such that `DP * TP * PP * EP == GPU count`, and `EP` divides
`num_experts`. On 8 GPUs with 64 experts:

| Layout | Placement | When it makes sense |
| --- | --- | --- |
| DP=1, TP=1, PP=1, EP=8 | 1 group of 8 ranks, 8 experts/rank | The reference case: experts dominate params, all-to-all stays in one node |
| DP=2, TP=1, PP=1, EP=4 | 2 groups of 4 ranks | All-to-all is node-local per half; more data parallelism but each rank holds 16 experts |
| DP=2, TP=2, PP=1, EP=2 | 2 DP x 2 TP groups | Dense part needs TP; experts sharded only 2-way (32/rank) |
| DP=1, TP=1, PP=2, EP=4 | 2 pipeline stages of 4 ranks | Very deep model where weights-per-rank must shrink further |

Layouts such as **DP=4 × EP=8** or **DP=8 × EP=8** are **infeasible on 8
GPUs**: the product (32 or 64) exceeds the device count. The four dims must
multiply to exactly the number of GPUs.

Reasoning checklist when choosing:

1. Does `EP` divide `num_experts`, and does `num_experts / ep` put a sane
   number of experts per rank?
2. Does the product `DP*TP*PP*EP` equal the GPU count?
3. Can the interconnect afford the per-MoE-layer all-to-all at this EP
   degree? (Fast intra-node links favor larger EP.)
4. Which component dominates memory — dense params (favor ZeRO/TP), expert
   params (favor larger EP), or activations (favor recompute or smaller
   micro-batch)?

## Summary table

| Scheme | Dim sharded | Comm pattern | When to use | Pitfalls |
| --- | --- | --- | --- | --- |
| DP | Batch | Gradient all-reduce | Any scale; the default | Model replicated → memory cost; ZeRO needed at scale |
| TP | Matmul weights | All-reduce/all-gather per op | Large d_model, fast intra-node links | Requires NVLink-class interconnect; awkward with EP tensors |
| PP | Layers | Point-to-point activations/gradients | Very deep models, many GPUs | Pipeline bubbles; activation-in-flight overhead |
| EP | Experts | **All-to-all per MoE layer** (dispatch + combine) | Expert-heavy MoE at scale | Must divide `num_experts`; all-to-all dominates; fp32 comm dtype |
