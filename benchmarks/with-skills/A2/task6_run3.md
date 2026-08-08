I'll start by exploring the workspace to find the skill resources (calculator, knowledge, examples) referenced in the skill.
The moe-architecture skill references `tools/moe_calculator.py`, `knowledge/`, and `examples/`. Let me locate those resources in the workspace.
I don't have shell/file access in this session to run `tools/moe_calculator.py`, so I'll compute every figure from the skill's exact formulas and note that the arithmetic follows the calculator's equations. Let me verify the arithmetic and finalize the design.
# 1B Dense-Equivalent MoE on a 25 Gbps Inter-Node Cluster — Architecture Document

**Design token:** `1B-moe-ep4-16x2`

---

## Overview

This design replaces a ~1B dense-equivalent model with a Mixture-of-Experts
variant whose **activated parameters ≈ 1.02B** (the compute/quality match to a
1B dense model) using **16 experts, Top-2 routing** and **total parameters of
6.31B**. The hard constraint is the interconnect: the cluster's inter-node
link is capped at **25 Gbps (no InfiniBand)**, and the expert-parallel degree
is **fixed at EP = 4** because the topology groups GPUs into pods of 4. No
published recipe targets this combination, so the design makes communication
containment its first-order objective:

- **EP = 4** expert groups are pinned *entirely within one pod of 4 GPUs*, so
  the all-to-all token dispatch never crosses the 25 Gbps inter-node link.
- The only cross-pod traffic per step is the **DP gradient all-reduce
  (≈ 4.73 GiB/step, ZeRO-2 sharded)**, which takes ~1.5 s at 25 Gbps and is
  overlapped with compute.
- The expert all-to-all token volume is **48 GiB per pod per step
  (192 GiB/step cluster-wide) and is carried by the fast intra-pod fabric,
  not the 25 Gbps link.**

Headline numbers: **total params 6,306,709,504 (6.31B)**, **activated params
1,021,886,464 (1.02B ≈ 1B dense-equivalent)**, **param ratio 9.79×**,
**num_experts = 16**, **top_k = 2**, **6.33 GFLOP/token**.

---

## Parameters table

Config: `num_layers = 12`, `d_model = 2048`, `ffn_mult = 2.5` (`ffn_dim =
5120`), `num_experts = 16`, `top_k = 2`, `vocab = 32000`, `seq_len = 2048`,
EP = 4. (Formulas per `tools/moe_calculator.py`; `EP = 4` divides `E = 16`.)

| Component | Dense | MoE | Activated |
| --- | --- | --- | --- |
| Attention + norms (all layers) | 201,375,744 | 201,375,744 | 201,375,744 |
| Expert FFNs (all layers) | 377,487,360 | 6,039,797,760 | 754,974,720 |
| Embedding | 65,536,000 | 65,536,000 | 65,536,000 |
| **Total** | **644,399,104** | **6,306,709,504** | **1,021,886,464** |
| | (0.64B) | **(6.31B)** | (1.02B ≈ 1B dense-equiv.) |

- per-expert FFN (GLU) = `3 × 2048 × 5120` = 31,457,280 params
- `param_ratio = 6,306,709,504 / 644,399,104` = **9.79×** (MoE vs dense baseline
  of this shape; total/activated = 6.2×, same order as Mixtral's 6.7×)
- FLOPs/token = `6 × activated + 4 × L × d × seq_len` = 6,131,318,784 +
  201,326,592 = **6,332,645,376 ≈ 6.33 GFLOP/token** (dense baseline of this
  shape: 4.07 GFLOP/token; ratio 1.56×)

---

## Routing

| Choice | Value | Justification |
| --- | --- | --- |
| Strategy | **Top-2** (learned linear router) | Default for training quality; blends two experts per token |
| num_experts | **16** | Typical production range (16–64); 16/EP4 = 4 experts/GPU, a healthy shard. **E > 16 (e.g. 64) adds only memory, never less all-to-all** — volume is `top_k × tokens`, independent of E — so larger E is a pure liability here |
| top_k | **2** | Top-1 would halve *intra-pod* dispatch, but that traffic is cheap; the quality gain of Top-2 is free against the 25 Gbps budget |
| Capacity factor | **1.0** | Zero padding waste = zero wasted dispatch. Safe only because the aux loss below enforces near-perfect balance |
| Aux loss | **0.01** | Upper end of 0.001–0.01: at capacity 1.0 any imbalance drops tokens, so balance is mandatory. Add router-logit jitter at train time |

Top-2 stays affordable specifically because the extra dispatch is **intra-pod**
(NVLink/PCIe, ≥ 32 GB/s), not on the 25 Gbps inter-node link.

---

## Training implications

### Cluster & parallelism

4 pods × 4 GPUs = 16 GPUs (H100/H800-class, 80 GB). **DP = 4, TP = 1, PP = 1,
EP = 4**, with `DP × TP × PP × EP = 16` and `EP | num_experts`. Each pod holds
one full replica with its 16 experts sharded 4-ways across the pod's GPUs.

- Batch geometry: `micro_batch = 8`, `grad_accum = 16`, `DP = 4` → global batch
  = **1,048,576 tokens/step** (512 × 2048). Tokens/expert/GPU/micro-batch =
  `8 × 2048 × 2 / (16/4)` = **8,192** (≥ 8–64 floor).
- Per-GPU memory ≈ 3.6 (params) + 3.6 (grads) + 18.9 (AdamW, ZeRO-1/2 across
  DP=4) + ~10 (activations) ≈ **36 GB on 80 GB** — ~55% headroom; BF16.

### Compute

16 × H100 bf16 ≈ 1.58×10¹⁶ FLOP/s peak; at ~40% MFU the 1.05M-token step
(`6.33e9 × 1.05e6 ≈ 6.6×10¹⁵` FLOP) runs in **~1.4–1.7 s wall**.

### Token-communication volume per step — the 25 Gbps budget

**All-to-all token dispatch (expert routing) — intra-pod only:**
per MoE layer, per pod, per step, `tokens × top_k × d_model × dtype_bytes` =
`262,144 × 2 × 2048 × 2` B = 2.0 GiB dispatch, 2.0 GiB receive → **4.0 GiB per
layer per pod**, × 12 layers = **48 GiB/pod/step**; cluster-wide = **192
GiB/step**. This traffic rides the **intra-pod fabric** (NVLink ≈ 0.1 s, or
PCIe Gen4 ≈ 1.5 s, per step) and **consumes 0 bytes of the 25 Gbps inter-node
link**, because EP groups never span pods.

**Inter-node traffic per step (the only thing on 25 Gbps):**
the DP gradient all-reduce across the 4 pods. Full gradients = 6.31B × 2 B =
12.61 GB; **ZeRO-2 sharding** cuts each rank's reduce-scatter + all-gather to
`2 × (3/4) × (12.61/4)` = **≈ 4.73 GiB per rank per step** → **~1.5 s at
25 Gbps (3.125 GB/s)**. This overlaps the ~1.4–1.7 s compute phase
(comm/compute overlap), so **link utilization is bounded and the step time is
set by compute, not interconnect**. No TP/PP across pods and no checkpoint
writes cross the link, so gradient sync is the *only* cross-pod consumer.

**Headroom lever:** at 1M tokens/step the sync window ≈ compute window. Raising
the global batch to 2M tokens/step or DP to 8 (32 GPUs, 2.36 GiB/rank/step →
~0.75 s) gives ≥ 30% margin. EP stays 4 by construction.

---

## Risks

1. **Param growth (6.31B total, 9.79× the dense shape, 6.2× activated).**
   Memory and checkpoint cost scale with total params while FLOPs stay flat.
   *Mitigation:* cap experts at 16 (4/GPU); never raise E chasing capacity —
   it cannot reduce communication.
2. **Cross-pod EP would blow the budget.** If experts were spread over pods,
   the 192 GiB/step all-to-all would need ~12+ minutes at 25 Gbps — fatal.
   *Mitigation:* hard rule — EP groups are pinned inside pods; cluster
   scheduler must not place an EP group across nodes.
3. **Link saturation during gradient sync.** Unsharded ring all-reduce moves
   19 GiB/rank/step → 6.1 s > step time. *Mitigation:* ZeRO-2 gradient
   sharding (4.73 GiB → 1.5 s) + overlap; escalate to ZeRO-3 if margin is
   needed.
4. **Token drop at capacity 1.0.** Any routing imbalance drops tokens with no
   overflow buffer. *Mitigation:* aux loss 0.01 + jitter; monitor per-expert
   counts; fall back to capacity 1.25 only if drops appear.
5. **Intra-pod fabric assumption.** The design relies on ≥ 32 GB/s inside a
   pod (48 GiB/step/pod must fit the step). *Mitigation:* require NVLink (or
   PCIe Gen4+ ×16) within pods; validate all-to-all time in profiling.
6. **Router collapse.** Strong aux loss (0.01) plus jitter; verify effective
   experts ≥ 8 in the first thousand steps.

---

*Validation note:* every parameter/FLOP figure above is computed with the
exact formulas of `tools/moe_calculator.py` (arithmetic carried out by hand in
this session — no shell access to execute the script); recompute with the tool
before locking the config. `EP = 4` divides `num_experts = 16`, and
`DP × TP × PP × EP = 4 × 1 × 1 × 4 = 16 = GPU count`.
Done. The design: **12 layers, d=2048, ffn_dim=5120, 16 experts, top-k 2, vocab 32000** → activated 1.02B (1B dense-equivalent), total 6.31B, 6.33 GFLOP/token.

Constraint satisfaction, in short:
- **EP = 4** (pods of 4 GPUs), `EP | num_experts` (16/4 = 4 experts/GPU), DP=4 → DP×EP = 16 GPUs.
- **Token-communication volume/step:** all-to-all = 4 GiB/layer/pod → **48 GiB/pod/step, 192 GiB/step cluster-wide**, but it's **intra-pod only** — EP groups are pinned inside pods, so it never touches the 25 Gbps link.
- The **only 25 Gbps traffic** is ZeRO-2-sharded DP gradient sync: **≈ 4.73 GiB/rank/step → ~1.5 s** at 3.125 GB/s, overlapped with the ~1.4–1.7 s compute step. Headroom via larger global batch or DP=8.

Note: no shell/file access in this session, so figures were computed by hand from the skill's exact formulas rather than executed via `tools/moe_calculator.py` — flagged in the doc to recompute before locking the config.