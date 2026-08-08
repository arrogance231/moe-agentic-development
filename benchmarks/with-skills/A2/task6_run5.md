## 1B Dense-Equivalent MoE — Architecture Document (EP=4, 25 Gbps Inter-node)

*All figures are computed with the exact formulas implemented in `tools/moe_calculator.py` (attention `4·d²`, expert GLU `3·d·f`, embedding `vocab·d`, `flops = 6·activated + 4·L·d·seq`).*

### Overview

The dense-equivalent baseline is **955,195,392 (~0.96B)** parameters: `num_layers=24`, `d_model=1536`, `ffn_mult=4` (`ffn_dim=6144`), `vocab=32000`, `seq_len=2048`. The MoE variant replaces every FFN with **16 experts, top-2** routing: **11,147,354,112 total params (~11.15B)** and **1,634,672,640 activated (~1.63B)**, for a parameter ratio of **11.67×** over dense and **1.71×** activated. The design's load-bearing constraint decision is the parallelization layout: **expert-parallel degree is fixed at EP=4**, and each EP group is placed inside one 4-GPU pod, so all all-to-all token dispatch rides **intra-pod NVLink** and the capped **25 Gbps inter-node link carries only ≈1.1 GB/step** of dense-parameter gradient all-reduce (0.35 s serialized), which fits a ~0.5 s step.

### Parameters table

| Component (24 layers) | Dense | MoE | Activated |
| --- | --- | --- | --- |
| Attention (`24 × 4·d²`) | 226,492,416 | 226,492,416 | 226,492,416 |
| Expert FFNs (`24 × E·3·d·f` / `24 × k·3·d·f`) | 679,477,248 | 10,871,635,968 | 1,358,954,496 |
| Layernorms (`24 × 2·d`) | 73,728 | 73,728 | 73,728 |
| Embedding (`32,000 × 1,536`) | 49,152,000 | 49,152,000 | 49,152,000 |
| Router (`≈24 × 16 × 1,536`, negligible) | — | 589,824 | 589,824 |
| **Total** | **955,195,392** | **11,147,354,112** | **1,634,672,640** |

**num_experts = 16, top_k = 2.** Total-parameter figure: **11,147,354,112** (≈11.15B). `param_ratio = 11.67`; activated/dense = 1.71.

### Routing

- **Strategy: Top-2, `top_k=2`.** Default for training-quality work; the extra expert's dispatch cost is paid on intra-pod NVLink, not the capped link, so there is no bandwidth reason to drop to Top-1.
- **Capacity factor = 1.25.** Absorbs routing imbalance without dropping tokens; at balanced load, per-expert demand is 4,096 tokens/micro-batch against 5,120 capacity → zero drops, 25% headroom under skew.
- **Auxiliary load-balancing loss = 0.01.** Top of the 0.001–0.01 band — with 16 experts and 4 ranks/pod, a weak aux loss risks collapse starving experts; the interconnect is cheap here, so over-strengthening is not a bandwidth concern.
- **Router logit jitter at train time.** Added to fight deterministic early specialization (per `moe-debugging`).

### Training implications

**Compute.** `flops_per_token = 10,110,025,728` (≈10.11 GFLOP/token) vs dense `6,033,162,240` (≈6.03) → **1.68×** dense. At a global batch of 128 (`micro_batch=4 × grad_accum=8 × dp=4`, `seq_len=2048` → 262,144 tokens/step) the step is **2.65 PFLOPs** ≈ 0.5 s at ~35% MFU on 16×H100.

**Memory (per GPU, bf16 + AdamW).** Experts sharded by EP=4: 2.72B expert + 0.28B dense = 2.99B params → 5.99 GB weights + 5.99 GB grads + 35.9 GB optimizer ≈ 48 GB, plus ≈6 GB activations → **≈54 GB/GPU**; fits H100-80GB with 33% headroom (does not fit 40GB-class GPUs — see Risks).

**Parallelization.** 16 GPUs = 4 pods × 4 GPUs; **DP=4 × EP=4**, product = 16 ✓; EP=4 divides num_experts=16 (4 experts/rank ✓).

**Interconnect constraint (25 Gbps, EP=4).**
- **Token-communication volume per step:** each MoE layer dispatches+receives `tokens × top_k × d_model × 2 B × 2 directions = 262,144 × 2 × 1,536 × 2 × 2 = 3,221,225,472 B ≈ 3.22 GB`; × 24 MoE layers = **77.3 GB/step**.
- **How it fits the 25 Gbps budget:** because EP=4 is fixed and each EP group maps to one pod of 4 GPUs, *all* of that 77.3 GB stays intra-pod on NVLink and **never crosses the 25 Gbps inter-node link**. The capped link only carries what must span pods: the data-parallel all-reduce of the dense (non-expert) params — 275,718,144 params × 2 B ≈ 0.55 GB, ~2× for ring all-reduce ≈ **1.1 GB/step**. Serialized at 25 Gbps (3.125 GB/s) that is **0.35 s < 0.5 s step**; with comm/compute overlap it occupies ~70% of one direction and is comfortably within budget. A naive design that let the all-to-all cross nodes would need 77.3 GB / 3.125 GB/s ≈ **24.7 s/step** — infeasible; pod-aligned EP placement is what makes this work.

### Risks

1. **Pod-alignment fragility (highest).** The design depends on EP groups never spanning pods. Any scheduler that splits an EP group across pods pushes the full 77.3 GB/step onto the 25 Gbps link (24.7 s/step). Mitigation: pin ranks to pods; validate placement before training.
2. **Over-parameterization.** 11.15B total params on 16 GPUs — feasible per the memory budget, but checkpoint size is ~12.6× the dense baseline. Mitigation: sharded checkpoints (ZeRO-style), bf16.
3. **Router collapse / expert imbalance.** 16 experts on 4 ranks with small per-layer throughput. Mitigation: aux loss 0.01, jitter, capacity 1.25, periodic `router_distribution.py` entropy checks.
4. **Memory on smaller GPUs.** ~54 GB/GPU fails on 40–48GB hardware. Mitigation: activation recomputation (activations → ~3 GB) and/or expert offload.
5. **Top-2 vs Top-1 at inference.** Top-2 doubles active-expert memory/throughput at serving time; acceptable for training, revisit for latency-bound inference.