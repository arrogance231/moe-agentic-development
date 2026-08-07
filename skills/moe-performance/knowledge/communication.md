# All-to-all and expert-parallel communication

When an MoE model is sharded with expert parallelism (EP), every token crosses
the interconnect at every MoE layer: it must be dispatched to the rank that
owns its chosen experts, and the expert outputs must be gathered back. This
traffic, not FLOPs, is often what caps MoE throughput. This note covers what
the traffic looks like, when it dominates, and how to reduce it.

## Expert parallel communication patterns

Under EP, each rank owns a disjoint subset of the experts
(`num_experts / ep` experts per rank). Each MoE layer runs two all-to-all
collectives:

1. **Dispatch** — the local router sends every token to the rank owning the
   expert(s) it selected (one send per `top_k` selection, to possibly
   different ranks).
2. **Receive / gather** — after the expert FFN, the host rank gathers each
   expert's outputs back and combines them (weighted by the router
   probabilities).

Volume per MoE layer, per step, is roughly:

```text
dispatch_volume ≈ top_k × tokens_per_step × dtype_bytes × (one send per token)
gather_volume   ≈ same, in the reverse direction
```

So total all-to-all bytes per step scale with `top_k × tokens_per_step ×
dtype_bytes`, times the number of MoE layers. With `top_k = 2`, bf16 (2 bytes),
and 2M tokens/step, each MoE layer moves about `2 × 2e6 × 2 = 8 MB` per
direction — which, multiplied by 24 layers, is non-trivial traffic per step.

All-to-all dominates when:

- **top-k is large** — every additional expert selection doubles nothing but
  adds linearly to dispatch volume.
- **EP degree is high** — the exchange involves more ranks, so each send is
  split across more destinations; per-message overhead and network contention
  grow.
- **The interconnect is slow** — NVLink keeps intra-node all-to-all cheap;
  InfiniBand (or worse, Ethernet) between nodes makes the same volume far more
  expensive.
- **Communication is not overlapped with compute** — blocking collectives turn
  the all-to-all into pure dead time.

## Token dispatch and capacity

The capacity factor decides how many tokens each expert is allowed to receive
relative to a perfectly balanced load:

```text
capacity_per_expert = capacity_factor × (tokens / num_experts)
```

- **Capacity factor > 1 (e.g. 1.25)** — experts can absorb more than the
  balanced share, so skewed routing rarely drops tokens, but every step still
  allocates the full padded buffers and the dispatch volume is set by the
  *capacity*, not the actual load. Imbalance shows up as idle expert slots, not
  drops.
- **Capacity factor = 1.0** — buffers and dispatch volume shrink to the
  balanced minimum, but any expert receiving more than its balanced share
  **drops** the overflow tokens (quality loss) unless the router is kept
  balanced.

Balanced dispatch (roughly equal token loads) fills the capacity exactly;
skewed dispatch either wastes capacity (padding) or drops tokens. Load
balancing is therefore a communication lever as much as a quality lever: a
balanced router lets you lower the capacity factor without incurring drops,
which cuts both compute and dispatch volume.

## Overlap strategies

The all-to-all does not have to be serialized with the FFN. Common patterns:

- **Compute/communication overlap** — dispatch of layer *L+1* is issued while
  the expert FFNs of layer *L* are still executing, so the interconnect and
  the tensor cores work in parallel. This is the single biggest latency win
  when communication-bound.
- **Delayed / async all-to-all** — the receive is posted early and only
  awaited right before the FFN input is needed, hiding interconnect latency
  behind other work.
- **Pipelining dispatch with FFN compute** — dispatch of the forward pass for
  a micro-batch is overlapped with the backward pass of the previous
  micro-batch inside a gradient-accumulation loop.

Software support is conceptual at this phase: DeepSpeed MoE (`deepspeed.moe`)
and Megatron-LM (`--num-experts` with `--expert-model-parallel-size`) both
expose EP plus communication overlap toggles; Megatron additionally lets you
overlap the all-to-all with the transformer compute via its MoE token-dispatch
implementation. The skill does not prescribe a specific framework's flags —
the training-setup phase (`moe-training`) produces the exact config.

## Reducing comm volume

Ordered from most to least intrusive:

- **Lower top-k** (top-2 → top-1) — halves dispatch volume. Cost: quality.
- **Lower capacity factor** (1.25 → 1.0) — shrinks buffers and dispatch volume
  toward the balanced minimum. Cost: dropped tokens under imbalance.
- **Fewer EP groups / lower EP degree** — fewer ranks participate in the
  exchange, cutting per-send fan-out. Cost: more experts per rank, more
  activation memory per rank.
- **Communication dtype** — dispatch in bf16 (or fp8) instead of fp32 roughly
  halves or quarters the bytes. Cost: precision in the routed sums (fp32
  accumulate where it matters).
- **Sequence packing** — fills capacity exactly, removing padding tokens that
  would otherwise be dispatched to experts and waste bandwidth on zeros.

## Measurement

From the metrics this skill collects, a communication-bound run looks like:

- **Step time that does not track FLOPs** — adding experts (same activated
  params) or raising top-k raises step time; raising micro-batch barely does.
- **High bubble time with busy GPUs** — `profilers/throughput_profiler.py`
  shows `bubble_pct` from the `busy` column; if GPUs are busy a small fraction
  of the step but MFU is not near the utilization ceiling, the gap is usually
  communication or kernel inefficiency rather than pure idle.
- **Low MFU despite decent utilization** — `MFU % (ROUGH EST.)` well below the
  `GPU util proxy` means the busy time is not spent on useful matmuls; a
  large share of that is all-to-all waiting.

To confirm, compare step time at EP=4 vs EP=8 on the same config: if step time
drops materially at higher EP, the run is communication-bound (the `moe-architect`
rule that EP cost is one all-to-all per MoE layer holds here). The concrete A/B
procedure lives in `examples/throughput-optimization.md`.
