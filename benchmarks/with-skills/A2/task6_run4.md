The skill's calculator tool isn't available in this session, so I applied its exact formulas by hand. Here is the architecture document.

---

# MoE Architecture: 1B Dense-Equivalent on a 25 Gbps Interconnect Cluster

## Overview

The dense-equivalent baseline is a **1.14B-parameter** model (16 layers, `d_model=2048`, FFN width `8192` (`ffn_mult=4`), `vocab=32000`, `seq_len=2048`). The MoE replaces each dense FFN with **32 experts** of width `4096` (`ffn_mult=2`) and activates **top-2** per token. Because 2 × per-expert FFN exactly equals the dense FFN width, **activated params = 1.14B = the dense baseline**, so the MoE is FLOPs-neutral versus dense at **7.10 GFLOPs/token**. Total params grow to **13,218,938,880 (13.22B)**, a **11.6×** ratio.

The binding constraint is the interconnect: **25 Gbps (3.125 GB/s) inter-node, expert-parallel degree fixed at EP=4**. The design neutralizes this by (a) placing each EP group wholly inside one 4-GPU pod so all MoE token all-to-all traffic is **intra-node**, and (b) keeping the only inter-node traffic — dense-parameter gradient all-reduce over DP=2 — at **670 MB/step ≈ 5% of step time**, leaving ~95% headroom. Total token-communication (all-to-all dispatch+combine) is **64 MiB (67.1 MB) per global step**.

## Parameters

| Component | Dense | MoE | Activated |
| --- | ---: | ---: | ---: |
| Attention (16 layers × 4·d²) | 268,435,456 | 268,435,456 | 268,435,456 |
| Expert FFNs (16 layers) | 805,306,368 | 12,884,901,888 | 805,306,368 |
| Embedding (32000 × 2048) | 65,536,000 | 65,536,000 | 65,536,000 |
| LayerNorm (16 × 2·d) | 65,536 | 65,536 | 65,536 |
| **Total** | **1,139,343,360** | **13,218,938,880** | **1,139,343,360** |

- Per-expert FFN (GLU) = `3·2048·4096 = 25,165,824`; MoE row = `16 × 32 × 25,165,824`; activated row = `16 × 2 × 25,165,824`.
- `num_experts = 32`, `top_k = 2`. Router = `32·2048·16 ≈ 1.05M` params, <0.01% of the total, omitted per the skill's convention.
- `param_ratio = 13,218,938,880 / 1,139,343,360 = 11.60`.
- `flops_per_token = 6 × 1,139,343,360 + 4 × 16 × 2048 × 2048 = 7,104,495,616 ≈ 7.10 GFLOPs/token` — identical for dense and MoE.

## Routing

- **Strategy: Top-2** — training-focused (not latency-bound inference); the skill's default for training, retaining expert knowledge blending that Top-1 sacrifices. The doubling of dispatch volume is acceptable because the volume is tiny (below).
- **top_k = 2** — chosen to keep activated params at the 1B target: 2 × 4096-width experts = one 8192-width dense FFN.
- **Capacity factor: 1.25** — absorbs routing imbalance without dropping tokens, the skill's recommended band (1.0–1.25); keeps all-to-all volume bounded at 1.25× ideal.
- **Auxiliary loss: 0.01** — top of the 0.001–0.01 band. With only 8 experts per rank and a low-bandwidth link, imbalance would pad expert buffers and inflate dispatch; the higher aux coefficient keeps load tight at modest routing-quality cost.

## Training implications

**Compute.** 7.10 GFLOPs/token, exactly equal to the dense baseline (activated = dense by construction). Expert count only moves total params, never activated params or FLOPs.

**Memory.** Total 13.22B vs activated 1.14B. With EP=4, each rank owns 8 of 32 experts; expert params/grads/optimizer states are local to the owning rank. The dense part (attention + LN + embedding + router ≈ 335.1M params) is replicated per DP rank. On 8 × 80 GB GPUs this fits with headroom; checkpoints are dominated by the expert weights.

**Parallelism (8 GPUs = 2 pods × 4).** `DP=2 × EP=4 × TP=1 × PP=1`; product = 8, and `EP=4` divides `num_experts` (32/4 = 8). One all-to-all pair per MoE layer. `micro_batch=4` × `seq=2048` × `grad_accum=32` × `DP=2` → **524,288 tokens/step** (~122K tokens/s at an estimated ~4.3 s/step at 35% MFU on 8×A100-80GB). Tokens per expert per GPU per micro-batch ≈ 2048, far above the 8–64 utilization floor.

### Interconnect budget (the constraint)

**Expert-parallel degree used: 4** (fixed, forced by the 4-GPU pod topology).

**Token-communication volume per step** (MoE all-to-all, dispatch + combine, bf16):
- Per MoE layer, per EP group: `262,144 tokens × top_k=2 × 2 bytes × 2 (dispatch+combine)` = **2.0 MiB/layer** (plus ~1% metadata for expert ids, negligible).
- × 16 layers = **32 MiB (33.6 MB)/step per EP group**; × DP=2 = **64 MiB (67.1 MB)/step globally**.

**How it stays within 25 Gbps (3.125 GB/s):**

| Traffic | Volume/step | Rides | Worst-case time at 25 Gbps |
| --- | --- | --- | --- |
| MoE all-to-all (dispatch+combine) | 64 MiB (67.1 MB) | **intra-node** (EP group = one pod) — never crosses the link | 21 ms (if it did cross) |
| Dense-gradient all-reduce (DP=2, bf16) | 670.2 MB | inter-node 25 Gbps link | 214 ms ≈ 5% of a ~4.3 s step |

The design keeps the 25 Gbps link nearly idle by four mechanisms:

1. **EP groups are co-located within a pod.** All all-to-all (the MoE-specific traffic, 64 MiB/step) rides intra-node fabric, so the 3.125 GB/s budget is not consumed by it at all. Even in the worst case where it did cross, it would cost only ~21 ms/step (<1%).
2. **Expert gradients never leave their node.** Each expert is fully owned by one GPU, so the bulk of parameters (12.88B expert params) produce zero inter-node traffic.
3. **Inter-node traffic is minimized by construction.** Only the 335.1M dense params are all-reduced, and DP=2 is the minimum replication. At bf16 that is 670.2 MB/step ≈ 0.21 s ≈ 5% of the estimated step, against a ~13.4 GB/step inter-node budget.
4. **Dispatch volume is kept small at the source:** top_k=2 (not 4), capacity factor 1.25 bounds per-expert buffers, and aux loss 0.01 keeps load balanced so dispatch never spikes from overflow/straggler padding.

## Risks

- **Over-parameterization without compute gain (high).** The 11.6× ratio buys no activated capacity or FLOPs; it is pure memory/checkpoint cost. Mitigation: EP=4 shards experts 4-way, the dense part is small, and 13.2B fits 8×80 GB; accepted as the price of a 32-expert mixture.
- **Placement regression on the 25 Gbps link (high).** If an EP group ever spans pods, 64 MiB/step of all-to-all crosses a 3.125 GB/s link. Still only ~21 ms/step, but the design's 5% figure depends on co-location; the launcher must pin EP ranks to one pod.
- **Routing collapse / imbalance under low aux loss.** With 8 experts/rank, collapse would starve gradients and pad buffers. Mitigation: aux loss 0.01 (top of band), capacity factor 1.25, optional router-logit jitter at train time; verify effective experts ≥ 16 per the debugging skill.
- **Token drops at capacity 1.25 under imbalance.** Mitigation: monitor overflow fraction; the 0.01 aux loss is sized to prevent it; raise capacity factor only if quality regresses.
- **Quality ceiling from per-expert size.** Each expert is 0.5× the dense FFN; if quality plateaus, the fix is more experts at the same compute — but that raises the total-parameter ratio further, so it must be gated on hardware memory.

---

**Constraint satisfaction summary:** EP=4 (fixed), token-communication = **64 MiB (67.1 MB)/step** (all-to-all, all 16 MoE layers), and the 25 Gbps budget is consumed only by the **670 MB/step dense-gradient all-reduce ≈ 5% of step time**, with all-to-all placed intra-pod so it never crosses the link.

Note: the skill's `tools/moe_calculator.py` was unavailable in this session; every figure above follows its documented formulas exactly and should be re-run through the calculator when the tool is accessible to confirm.