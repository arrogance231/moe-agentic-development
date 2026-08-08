I don't have file-write tools in this session, so I'll deliver the decision document directly in my reply. Let me first verify the literature facts I'm weighing so the citations are accurate.
Verified the key literature anchors. Here is the decision document.

---

# MoE Parameter Decision — 8×80GB (standard interconnect), 1B dense-equivalent

| | | |
|---|---|---|
| **Budget** | 8× 80GB GPUs, non-full-mesh interconnect (PCIe Gen4/5 or partial NVLink pairs) | |
| **Target** | Dense-equivalent 1B parameters (≈1B active params/token) | |
| **Date** | 2026-08-08 | |

## Executive summary

| Parameter | Chosen value | Convention |
|---|---|---|
| **Top-k routing** | **2** | token-choice, softmax gating |
| **Capacity factor** | **1.25** | Megatron/NVIDIA convention: `capacity = CF × k × (tokens/experts)`; equivalently **2.5** in the GShard/Switch (top-1-normalized) convention → **25% padding** above balanced load |
| **Auxiliary-loss coefficient** | **1e-3** | on the Switch/GShard `f·P` load-balance loss (+ optional router z-loss at 1e-4) |

The three choices are one internally consistent configuration: **top-2 routing for quality, 25% capacity headroom to make token-dropping essentially impossible, and a light auxiliary loss (1e-3) that is only required to keep imbalance from consuming that headroom.** All three parameters are set by the same underlying fact: at 1B dense-equivalent scale on 8 GPUs, **expert padding and all-to-all traffic are cheap relative to total step cost, so the system should spend compute to protect training quality rather than squeeze it for efficiency.** This is the opposite regime from the trillion-parameter settings where most capacity-factor guidance was derived.

## 0. What the budget actually binds

A dense-equivalent 1B MoE needs only ~16 layers, d_model ≈ 2048, 32–64 experts with per-expert FFN width ≈ 11k (active ≈ 268M attention + 738M expert params ≈ 1.0B; total params a multiple of that). Total model + AdamW optimizer + activations fit in 640GB HBM with ZeRO/expert-parallel **without memory being a constraint**. The two real constraints are:

- **Expert compute padding** — wasted FLOPs from routing to slots that stay empty, scaling with capacity factor.
- **All-to-all communication** — per MoE layer ≈ `tokens × k × d_model × 2 bytes` per dispatch; at ~4k tokens/micro-batch, top-2 ≈ 34MB per dispatch, ~0.5–1GB/step across 8–16 MoE layers. A few ms per step on Gen5/partial-NVLink: **real but not dominant at this scale.** This is why we stop at k=2 but are not forced down to k=1.

## 1. Top-k routing: chosen **2**

**Positions weighed.**
- *Top-1 (Switch Transformer, ST-MoE).* Simplest router; half the expert FLOPs and half the all-to-all volume per token; load balancing is easier because each token consumes exactly one slot; ST-MoE showed top-1 + CF 1.25 + aux 1e-2 achieves strong quality per FLOP. Best when communication or expert compute is the hard bottleneck.
- *Top-2 (GShard, Mixtral 8x7B, NVIDIA Megatron Mixtral recipe).* Two experts consulted per token → higher quality at equal active-parameter count, smoother gradients (two experts receive signal per token), production-validated at 8×7B scale. Costs 2× expert compute and 2× communication per token.

**Justification.** On 8 GPUs at 1B active params, the second expert is ~700M FLOPs of a step that is dominated by many other operations; the marginal all-to-all cost is ~34MB per dispatch — well within the interconnect budget (see §0). The quality and sample-efficiency gain of top-2 is therefore "free" at this scale, and it directly matches the two most relevant production precedents (Mixtral, Megatron Mixtral). DeepSeek's k=8 sigmoid-gating regime is a different routing family and not adopted here.

## 2. Capacity factor: chosen **1.25** (Megatron convention; 25% padding)

**Convention note (why the literature range 1.0–2.0 is confusing).** Frameworks define capacity differently:
- GShard/Switch/T5X/HF: `capacity = CF × tokens/experts` (top-1-normalized).
- Megatron/NVIDIA: `capacity = CF × k × tokens/experts` (top-k included).

The same physical buffer is CF=1.25 under Megatron and CF=2.5 under GShard for top-2. The physically meaningful quantity is **padding above the balanced load**, which is what I state below.

**Positions weighed.**
- *CF low (Switch CF=1.0 top-1; NVIDIA Mixtral CF=1.0 top-2).* Zero padding — every expert slot is used at perfect balance. Maximizes hardware efficiency; requires a strong aux loss (1e-2) to hold balance, and still drops tokens on imbalance. Switch reports this as the *speed*-optimal regime at trillion-param scale.
- *CF middle (ST-MoE 1.25; GShard 1.25 top-1-normalized).* ST-MoE's ablations found raising CF from 1.0→1.25 reduces token dropping and improves quality with acceptable cost; dropping tokens was shown to be more damaging than the equivalent padding.
- *CF high (GLaM-era top-2 configs).* More headroom but scales both padding FLOPs and communication; wasteful once aux loss already controls imbalance.

**Justification.** Chosen **25% padding (Megatron CF=1.25)** for three reasons. (1) At ~4k tokens/step and 64 experts, balanced top-2 load is only ~128 tokens/expert; routing noise at these small counts makes imbalance proportionally large, so a zero-padding configuration (Megatron CF=1.0) risks real dropping. (2) Padding cost is a small fraction of total step FLOPs at this scale, so the ST-MoE argument — spend capacity to avoid dropping — is affordable. (3) 25% headroom is enough that the aux loss only needs to keep imbalance from overflowing the buffer, which is what lets us run a light 1e-3 coefficient (next section). I explicitly reject the GShard-style 1.25-in-the-other-convention (62.5% coverage, ~37% expected dropping) and the 2.0 zero-padding end, both because they sit at the dropping/waste extremes this config is trying to avoid.

## 3. Auxiliary-loss coefficient: chosen **1e-3**

**Positions weighed** (range spans ~2 orders of magnitude plus the two anchors):
- *~1e-4 (GShard).* Light touch; viable only with large capacity headroom and huge production batches whose per-expert counts self-average. Too weak for our small per-step counts.
- *1e-2 (Switch; ST-MoE; NVIDIA Megatron Mixtral recipe).* The dominant production choice, but it is calibrated for **zero-padding** configurations: with CF=1.0 (either convention) imbalance converts directly into dropped tokens, so balance must be near-perfect and the loss must be strong. Switch explicitly notes the coefficient must be "small enough not to overwhelm the primary objective" — a caveat that matters more at 1B scale where a 1e-2 balancing term is proportionally heavier than at 1T scale.
- *0 (Mixtral release).* Proven at 8×7B but on large batches; without it, our small-batch imbalance would bleed into dropping.
- *HF production default: 1e-3.*

**Justification.** Because CF=1.25 already provides 25% headroom, the aux loss does not need to enforce near-uniform routing — it only needs to keep the worst-case expert within 25% of the mean. 1e-3 is strong enough to do that on noisy, small per-expert counts (unlike GShard's 1e-4) while weak enough to avoid degrading expert specialization (the known cost of over-weighted balancing losses). It also matches the transformers production default. A router **z-loss at 1e-4** (ST-MoE) is added as a stability adjunct against logit overflow in bf16, but it is auxiliary to this decision.

## 4. Internal consistency check

The three values are coupled: top-2 doubles expected per-expert load, which raises the drop risk that CF=1.25 absorbs; the 25% headroom in turn is what licenses the weak 1e-3 loss instead of Switch's 1e-2. Change any one and the others should move: e.g., the NVIDIA Mixtral pairing (CF=1.0, aux=1e-2) is the correct *same-spirit* configuration at zero padding, and would be our fallback if padding cost ever mattered.

## 5. Revision conditions

| Trigger | Change |
|---|---|
| Measured all-to-all > ~15–20% of step time (interconnect worse than assumed) | k=2→**1**, CF→1.0 (Megatron), aux→**1e-2** (Switch-style communication-bound config) |
| Logged token-drop rate > 1–2% per layer (Switch's budget is <1%) | CF→**1.5**, and/or aux→**1e-2** |
| Validation loss degraded by over-balancing (router too uniform, lost specialization) | aux→**1e-4** (GShard-style), keep CF 1.25–1.5 |
| Dead experts (some experts get ≈0 tokens over long windows despite aux) | aux→**1e-2**, add/adjust z-loss |
| Scale-up to ≥8B dense-equivalent, or batches large enough that per-expert counts self-average | CF→**1.0**, aux→**1e-2** (Megatron Mixtral recipe) |
| Eval-time compute cap needed | Keep training config; lower CF at inference (GLaM practice) |
| Switch to no-dropping gating (DeepSeek-style sigmoid, fixed top-k) | aux→0, capacity→unbounded (no-drop) |

## 6. References

- Lepikhin et al., *GShard* (arXiv:2006.16668) — top-2, capacity = O(N/E), aux loss with small multiplier (~1e-4).
- Fedus et al., *Switch Transformers* (JMLR 23, 2022) — top-1, CF=1.0, aux=1e-2 (swept 1e-1–1e-5); dropping <1%.
- Zoph et al., *ST-MoE* (arXiv:2202.08906) — top-1, CF=1.25, aux=1e-2 + z-loss 1e-4; dropping worse than padding.
- Jiang et al., *Mixtral of Experts* (arXiv:2401.04088) — top-2, no load-balancing loss.
- NVIDIA Megatron-LM docs & Mixtral recipe — capacity = `tokens_per_rank × topk × CF / experts`; top-2, CF=1.0, aux=1e-2.
- Hugging Face transformers defaults — router aux loss 1e-3, z-loss 1e-3.

---

Want me to save this into the repo (e.g., `benchmarks/harness/configs/A0/DECISION.md`) — note I don't have file-write tools in this session, so I'd need edit access enabled.