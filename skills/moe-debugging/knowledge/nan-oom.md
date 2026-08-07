# Numerical instability and memory failures

How exploding loss, NaN, OOM, and communication failures show up in MoE
training, and how to diagnose each from the logs and the analyzers.

## Exploding loss and NaN

**Mechanisms.** Loss blows up or goes to NaN through four common routes:

- **FP16 overflow.** Half precision has a narrow exponent range; a large
  forward pass (deep stack, long sequence) overflows to Inf and then to NaN.
  This is the classic MoE contributor because expert FFNs add many large
  matmuls per token.
- **LR too high / warmup too short.** The optimizer takes large steps before
  the router has stabilized, so a loss spike can compound into divergence.
- **Gradient explosion.** Without clipping, a large-gradient layer — often
  the router or a single over-loaded expert — injects a huge update that
  corrupts weights.
- **Loss spike → expert-count spikes.** A routing burst can dump an unusual
  number of tokens onto one expert in one step; the expert's gradient spikes
  and the loss follows. These spikes are learnable: they recur at the same
  steps as the router distribution changes.

**Evidence.** Run `analyzers/loss_analyzer.py` on the `step,loss` curve:
- `NAN` / `INF` counts tell you if the curve actually broke.
- `SPIKE` reports steps where a point departs its rolling median by more than
  5× the median absolute deviation, with the top-5 spike steps listed — the
  steps to correlate with expert token counts.
- `DIVERGENCE` fires when the last 20 steps are monotonic non-decreasing:
  the curve is running away, not plateauing.
- `PLATEAU` firing *without* spikes points to a capacity/collapse problem
  rather than a numerical one — see `collapse-imbalance.md`.

Cross-reference the spike steps with per-expert token counts from
`analyzers/router_distribution.py`; if spikes align with expert-count
spikes, the cause is routing burstiness rather than the optimizer.

**Fixes.** Gradient clipping at norm 1.0; a proper LR schedule with enough
warmup and decay; switch to BF16 (same training dynamics as FP32, no overflow
in the mantissa) or keep FP32 master weights; and if spikes correlate with
expert-count bursts, add per-expert load control (see the
`### Exploding loss / NaN` workflow in SKILL.md).

**Debugging flow.** 1) Run `loss_analyzer.py`, note which flags fire. 2) If
`NAN`/`INF`/`SPIKE`, check precision and clipping first — these are the
highest-probability causes. 3) Correlate spike steps with router statistics.
4) Apply the ranked fixes and confirm with the clip-on/off ablation over 100
steps.

## Out of memory

**Memory sources for MoE.** The footprint is a sum over distinct buffers:

- **Parameters.** The full expert count lives on the devices. `moe-training`
  parameter counts feed this term; at large expert counts the expert weights
  dominate and must be sharded (expert parallelism).
- **Optimizer state.** Adam-class optimizers keep 2–3 states per parameter
  (often BF16 weights + FP32 master + moments). This is the biggest term at
  scale and is what ZeRO-3/FSDP shards.
- **Activations.** Per-token activations held for backprop. MoE activations
  scale with the *activated* (top-k) budget, not the full expert count.
- **Expert buffers.** The dispatched token buffers per expert. These scale
  with `capacity = ceil(capacity_factor * tokens_per_batch / num_experts)`,
  so the capacity factor directly inflates this term.
- **All-to-all buffers.** The dispatch/combine staging buffers for expert
  parallelism, sized by top-k × tokens × hidden width.

**Capacity-factor impact.** Raising the capacity factor to absorb imbalance
(linearly) raises the per-expert activation and dispatch buffers. The same
load can be absorbed by a load-balancing loss instead, letting the capacity
factor stay near 1.0 — the memory-efficient combination.

**Fixes, in order of leverage.** Reduce the capacity factor toward 1.0; enable
gradient checkpointing (recompute activations in backprop instead of storing
them); increase expert parallelism so expert weights and buffers shard across
GPUs; reduce the micro-batch size (or sequence length); and, as a last
resort, offload expert weights/optimizer state to CPU.

**Estimating.** Budget per device as:
`parameters × bytes + optimizer_state + activations (activated budget ×
top-k × capacity) + expert buffers + all-to-all buffers`, then compare against
HBM. Run `skills/moe-training/tools/memory_estimator.py` with your parameter
counts and parallelism layout (DP/TP/PP/EP), precision, optimizer, micro-batch,
sequence length, and `--recompute` for gradient checkpointing; it prints the
per-GPU breakdown (parameters, gradients, optimizer states, activations,
overhead) against the `--gpu-mem-gb` limit. The `skills/moe-architecture`
`tools/moe_calculator.py` provides the total/activated parameter counts that
feed the estimate. The failing step number tells you where the memory cliff
is — OOM early in training is usually weight/optimizer; OOM later as
sequence/batch grows is usually activations.

## Communication failure signatures

Communication failures look different in the logs than compute failures:

- **Step-time jumps.** A normal step takes X ms; intermittently it takes 2–5×
  as long. Look for a fixed pattern (every EP round, every all-to-all) rather
  than random jitter.
- **Hang.** A step never completes; the log stalls at the same point — often
  mid all-to-all with no loss printed.
- **OOM in comm buffers.** An OOM trace that points at dispatch/combine
  staging buffers rather than parameter or activation memory.
- **Dropped tokens as a proxy.** If the router is imbalanced, dispatch volume
  concentrates on a few devices; the straggler GPU paces every step. A rising
  `OVERFLOW` flag alongside slower steps points at dispatch imbalance.

Confirm with `analyzers/router_distribution.py` (is dispatch imbalanced?) and
`analyzers/expert_utilization.py` (is capacity inflated by the capacity
factor?), then apply the `### Communication bottleneck` workflow in SKILL.md:
reduce capacity factor, raise EP degree, overlap communication with compute,
and verify with an EP=8 vs EP=4 step-time comparison.
