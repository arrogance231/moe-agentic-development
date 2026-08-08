Grounding confirmed: published MoE recipes (Mixtral 8×7B: E=8, top_k=2; DeepSeek-V3/Qwen-3 EP groups) assume InfiniBand-class fabrics (200–400 Gbps/GPU), and the EP all-to-all is the dominant training bottleneck — exactly what must be engineered around here. Note also that published observability work confirms EP=4 spans a single node, i.e., it can be made NVLink-only. That observation is the load-bearing trick of the design below.

# MoE Architecture for a 1B Dense-Equivalent Model on 25 Gbps Interconnect (Fixed EP=4)

## 1. Overview

**Design goal:** a ~1B *dense-equivalent* (FLOPs-per-token) Mixture-of-Experts LM that trains efficiently on a cluster whose **inter-node links are capped at 25 Gbps (3.125 GB/s)** and where **expert-parallel degree is pinned at 4** because the topology groups GPUs into fixed 4-GPU pods.

**Central insight — make the fixed EP group *identical* to the network pod.** The constraint "EP must be 4" is turned from a handicap into an advantage by co-locating the entire expert set inside one 4-GPU node: **all 8 experts are replicated on every 4-GPU node (2 experts per rank), so the token all-to-all is pure intra-node NVLink traffic and contributes exactly 0 bytes to the 25 Gbps inter-node fabric.** Inter-node traffic is reduced to the *only* thing that must cross it: the periodic gradient all-reduce, which is amortized with gradient accumulation so that average link utilization stays under 25 Gbps.

**Why no published recipe covers this:** every open MoE training run (Mixtral, DeepSeek-V3, Qwen-3, Grok, Phi-3.5-MoE) assumes either (a) InfiniBand-scale fabrics (200–400 Gbps/GPU) that absorb the all-to-all, or (b) an EP degree equal to the full GPU count, scaling across nodes. This design takes the *opposite* route: it fixes the routing-all-to-all at pod scope by construction and budget-enforces everything else. This combination (25 Gbps scale-out + small fixed EP=4 + dense-equiv 1B) has no published precedent, so the doc is explicit about the numbers that make it work.

**Non-negotiable constraints honored:**
- Expert-parallel degree **= 4** (all ranks of a 4-GPU pod participate; each rank hosts 2 of the 8 experts).
- All inter-node token routing traffic = **0 bytes/step** by placement.
- Inter-node gradient traffic kept within the 25 Gbps envelope by gradient accumulation (K ≥ 16) and optional FP8 communication.

## 2. Parameters

Reference model: SwiGLU decoder-only, tied embeddings, bf16.

| Component | Formula | Params |
|---|---|---|
| Input/output embeddings (tied) | 50,000 × 2,048 | 102,400,000 |
| Attention per layer (Q,K,V,O) | 4 × 2,048² | 16,777,216 |
| Expert FFN (SwiGLU) per expert | 2 × 2,048 × 2,560 | 10,485,760 |
| Experts per layer | 8 × 10,485,760 | 83,886,080 |
| Non-expert total | 102,400,000 + 24 × 16,777,216 | 505,053,184 |
| Expert total | 24 × 83,886,080 | 2,013,265,920 |
| **Total parameters** | | **2,518,319,104 (≈ 2.52B)** |
| **Dense-equivalent (FLOPs per token)** | 505,053,184 + (top_k/E) × 2,013,265,920 = 505,053,184 + 503,316,480 | **1,008,369,664 (≈ 1.008B)** |

Hyperparameters: `d_model = 2048`, `n_layers = 24`, `n_heads = 16` (head_dim 128), `expert_d_ff = 2560`, `vocab = 50,000`, `seq_len = 4096`, SwiGLU + RMSNorm + RoPE.

| MoE-specific parameter | Value |
|---|---|
| **num_experts (E)** | **8** |
| **top_k** | **2** |
| **Expert-parallel degree (EP)** | **4** (fixed by constraint; 2 experts per rank) |
| Data-parallel degree (DP) | number of 4-GPU pods (reference config: 16 pods / 64 GPUs) |
| Routing | token-choice softmax top-2, aux load-balance loss (coef 1e-2) + router z-loss (1e-3), capacity factor 1.1 |

Why these numbers hit the 1B target: with top_k=2 and E=8, exactly 2/8 = 1/4 of expert parameters participate per token, so `dense-equiv = base (0.505B) + ¼ × experts (2.013B) = 1.008B`. The design is Mixtral-shaped (E=8, top_k=2) scaled down to the 1B-dense-equiv budget.

## 3. Routing choice and justification

**Choice: token-choice (Mixtral-style) `softmax(Top2(x·W_g))` with aux load-balancing loss + router z-loss; capacity factor 1.1; a small shared expert is deliberately omitted to keep the all-to-all exactly symmetric.**

- **top_k = 2, E = 8 gives dense-equivalent ≈ 1.008B.** top_k=1 (Switch-style) would drop effective compute to ~0.75B, below target, and hurt expert specialization; top_k=3–4 would push FLOPs per token past 1B. top_k=2 with 8 experts is the point that lands on target with proven quality.
- **No inter-node coordination cost.** Routing is a per-token local decision inside the node; with all 8 experts local, no global dispatch is needed, so the router's irregularity never touches the 25 Gbps link.
- **Load-balancing aux loss is mandatory, not optional.** With EP=4 inside one pod and NVLink dispatch, imbalance does not overflow a slow fabric (it just idles a GPU), but it directly reduces MFU. The aux loss (coef ~1e-2) plus z-loss keeps per-expert load within the 1.1 capacity factor so token dropping stays < 5%.
- **Rejected alternative — expert-choice routing (like DeepSeek):** expert-choice picks its own top tokens and is robust to collapse, but it requires a *global* token pool per expert to select from, which in EP=4 would force inter-node coordination or a cross-pod shuffle. That reintroduces the fabric traffic we eliminated. Token-choice with an aux loss achieves the same balance at zero inter-node cost.
- **Rejected alternative — shared expert (DeepSeek-V3 style):** a shared expert would run on every node (fine) but complicates the load-balance accounting and adds FLOPs per token (~4% here). Omitted to keep FLOPs exactly at 1.008B; can be added as a fallback if quality is short.

## 4. Training implications

**Communication architecture.** DP across pods (each pod a full model replica), EP=4 inside the pod. There are exactly two traffic classes:

**A. Token communication (all-to-all dispatch/combine) — 100% intra-node.**
Per-token dispatch bytes = `2 bytes × d_model × top_k = 8,192 B`; round trip (dispatch + combine) = `16,384 B/token`.

> **Token-communication volume per step (reference config: 16 pods × 4 GPUs = 64 GPUs; microbatch 65,536 tokens/pod; gradient-accum K = 16 → 16,777,216 tokens/optimizer step):**
> `16,777,216 × 16,384 B = 274,877,906,944 B ≈ 274.9 GB/step (256 GiB)` — **all carried on the intra-node NVLink fabric (~450 GB/s/GPU-class), ≈ 0.6 s/step overlapped with compute. Inter-node token volume = 0 B/step.**

Contrast: if the same EP=4 all-to-all had to cross 25 Gbps nodes, 274.9 GB would take **88 s/step** — impossible. Replicating experts per pod is what keeps dispatch off the fabric.

**B. Gradient synchronization — the only inter-node traffic.**
Model = 2,518,319,104 params → 5.04 GB (bf16). Ring all-reduce per pod per sync ≈ `2 × 5.04 × (N−1)/N`:
- N=16 pods: `9.44 GB` → `9.44 GB / 3.125 GB/s = 3.02 s` at full link speed.

**Budget check:** with K=16 accumulated microbatches, compute per optimizer step = 16.78M tokens × ~4.03 GFLOP/token ≈ 67.6 PFLOP → 3.4 s at peak (~6.2 s at 55% MFU). The 3.02 s sync is issued once per step and overlapped with the next step's forward/backward. **Peak burst = 25 Gbps (3.02 s); average utilization ≈ 9.44 GB / ~6.2 s ≈ 1.5 GB/s ≈ 12 Gbps < 25 Gbps cap. Token dispatch contributes 0 Gbps.** The envelope holds with ~2× headroom.

**Why K=16 works and small K doesn't:** a naive sync-per-microbatch (K=1) would inject 9.44 GB every ~0.4 s of compute → 25+ Gbps *before* reaching the compute-bound limit; K≥16 amortizes the 3.02 s sync into ≥6 s of compute. If the cluster grows to N pods, bytes scale as (N−1)/N (asymptotically 2×P = 10.1 GB) and K must be raised accordingly; FP8 communication halves bytes if needed.

**Memory and optimizer.** All 2.52B params resident per pod: ~5 GB (bf16) + 10 GB (FP32 masters) + 2× Adam moments — trivially fits 4 × 80 GB. No ZeRO required, which is *good* here: ZeRO's reduce-scatter/all-gather would add inter-node traffic and its shard-size is exactly what we're avoiding.

**Training setup.** bf16 with FP32 master weights; AdamW, warmup 2k steps, cosine decay; large effective batch (16.8M tokens/step) for MoE stability; activation checkpointing; load-balance aux loss coefficient 1e-2 annealed late in training; monitor per-expert utilization on every pod (they should all match since each pod sees an i.i.d. slice of the same data distribution). Pre-validate at 1 pod for 2–3k steps before scaling to N pods.

## 5. Risks

1. **Gradient sync is the hard ceiling.** A 2.52B-param model's all-reduce is 9.4–10.1 GB no matter what; at 25 Gbps that is ~3 s per sync. If the pod count grows past ~16–32, average utilization approaches the cap and MFU collapses unless K is raised or FP8 comm is enabled. There is no way to shrink bytes below model size without sharding experts — which would break the zero-dispatch property. **Mitigation:** hierarchical reduce (intra-pod NVLink reduce-scatter, then inter-pod) for better overlap; FP8 communication as a halving lever; keep pods ≥ 16 for training runs.
2. **Expert replication is memory-profligate, not FLOP-profligate.** Experts are duplicated on every pod (2.52B vs. 1.008B dense-equivalent stored). Compute is unchanged by design; memory is cheap here (80 GB cards) but this caps how far the model can grow before per-pod memory binds. Fixed EP=4 forbids the usual escape (shard experts across nodes).
3. **Router collapse still hurts, differently than in published clusters.** In InfiniBand clusters, imbalance → fabric congestion; here imbalance → idle GPUs and wasted NVLink bursts. Same remedy (aux + z-loss, capacity 1.1, drop <5%), same failure mode if annealing the aux loss too early. Large-batch MoE instability (the classic "every token hits the same 2 experts" collapse) is the primary convergence risk and is unproven at this scale with no published baseline.
4. **No published recipe = no off-the-shelf kernel/tuning path.** Megatron/SMP `expert_parallel_degree=4` exists but assumes InfiniBand/RDMA all-to-all; here the all-to-all must be *disabled across pods* and replaced with per-pod-local routing, which most frameworks won't do for you. Expect custom dispatch code (a `num_local_experts=8` + EP=4 placement) and no community-tuned hyperparameters. **Mitigation:** port Mixtral's stability kit (jitter, z-loss, top-2) and budget real time for a 1-pod bakeoff.
5. **Topology assumption is load-bearing.** The entire budget argument assumes a 4-GPU pod == one node with NVLink-class intra-pod fabric (as confirmed by published EP=4 observability work, which shows EP=4 has zero IB hops). If a "pod of 4" turns out to be 4 GPUs *inside* an 8-GPU node split across PCIe, the 274.9 GB/step dispatch becomes the bottleneck and the design must be re-derived. **Mitigation:** measure intra-pod bandwidth first; gate the go/no-go on that single number.
6. **Scaling is capped by both axes.** Cannot grow EP (fixed at 4), cannot grow per-pod model size beyond memory, and adding pods increases gradient bytes. This design is therefore a *fixed-envelope* solution: ~1B dense-equiv, 16–32 pods, 25 Gbps. It is intentionally not future-proof; the doc's value is making the trade-off explicit and budget-verifiable rather than accidental.

---

**One-line summary of constraint satisfaction:** EP = **4** (pod-co-located, 2 experts/rank); token-communication volume = **274.9 GB/step**, of which **0 B crosses the 25 Gbps interconnect** (all on NVLink by construction); the only fabric load is a **9.44 GB/step gradient all-reduce amortized to ≈12 Gbps average vs. the 25 Gbps cap**.