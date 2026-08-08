Based on my research across the primary literature (Switch Transformer, GShard, ST-MoE, Expert Choice, Mixtral, OLMoE, DeepSeek-MoE/V3), here is the decision document.

---

# MoE Hyperparameter Decision Record
## Budget: 8× 80GB GPUs (standard interconnect) · Target: 1B dense-equivalent

**Scope and assumptions.** I interpret "dense-equivalent 1B" as *active/FLOP-equivalent* parameters ≈ 1B (the standard way MoE models are compared: Mixtral 13B active/47B total, OLMoE 1B active/7B total). Under either reading the conclusions below are unchanged, because memory is not the binding constraint: 640 GB VRAM dwarfs a ~1–7B-parameter model. The binding constraints are (a) **all-to-all dispatch/combine bandwidth** over non-NVLink-full-mesh interconnect, and (b) wall-clock step time for a small model where communication is a large fraction of per-step cost. ST-MoE states the governing rule directly: *"if all2all/allreduce communications are slow, smaller capacity factors may dominate"* (Zoph et al., 2022, §7).

I assume a standard configuration: decoder-only transformer, an MoE layer replacing each FFN, **E = 8 experts** (one expert per device for expert-parallelism, per ST-MoE's "at most one expert per core" guidance for comm-bound systems), and capacity defined in the Switch convention below.

### Decision summary

| Parameter | **Chosen value** | Literature range | Basis |
|---|---|---|---|
| **Capacity factor (CF)** | **1.25** (train); 2.0 at eval | 1.0 – 2.0 | ST-MoE/Switch Pareto point; 1.0 drops too many tokens, 2.0 costs +7–14% step time |
| **Auxiliary-loss coefficient (α)** | **0.01** | 0.001 – 0.1 | Switch's swept optimum; confirmed by ST-MoE, OLMoE, Megatron default |
| **Top-k routing** | **1** (top-1) | 1 vs 2 in production | Halves all2all volume for a 1B-class model on slow interconnect; ST-MoE's top-2 gain is ~1/20th of a dense-tripling at 100B+ tokens |

**Capacity-factor convention (critical).** Two incompatible conventions circulate:
- **Top-1 (Switch) convention:** `capacity = ⌊(T / E) × CF⌋`, where T = tokens, E = experts.
- **Top-2 (GShard/Megatron) convention:** `capacity = ⌊(k·T / E) × CF⌋`.

GShard's "CF 2.0" and Mixtral's "CF 2.0" are *not* comparable to Switch's "CF 1.25" without normalization: total tokens processed per layer = `E × capacity`. Under top-1 CF 1.25, each layer processes `1.25·T` token-slots. A top-2 model with the same compute/communication footprint needs **CF = 0.625 in the Megatron convention** (i.e., `1.25·T` slots for `2·T` dispatch demands — no droop headroom at all). This is precisely why the convention must be stated whenever comparing published numbers. All values below are in the **top-1 (Switch) convention** for the chosen top-1 router.

---

## 1. Capacity factor — chosen: **1.25**

**Conflicting positions weighed:**
- **CF = 1.0 (Switch Transformer).** Fedus et al. (2021) show Switch models "perform better at lower capacity factors (1.0, 1.25)" and use 1.0 as default, arguing capacity is memory-scarce in the large-model regime. Token drops stay <1% *provided* the aux loss keeps load balanced.
- **CF = 1.25 (ST-MoE).** Zoph et al. (2022) find that for top-1, increasing train CF from 1.0 → 1.25 yields **+0.011** neg-log-perp (~1/10th of a dense model-tripling), and CF 0.75 produces 10.6% dropped tokens vs 0.3% at 1.25. They adopt 1.25 for training and 2.0 for eval.
- **CF = 2.0 (GShard, Mixtral, EC).** These are top-2 systems or expert-choice systems where CF 2.0 merely matches the per-token compute of top-2 routing; EC (Zhou et al., 2022) notes token-choice must overprovision **2–8×** to avoid drops. Under standard interconnect, GShard-style 2.0 is the expensive extreme.

**Justification for 1.25:**
1. **It is the Pareto-efficient point, and here the scarce resource is interconnect, not memory.** Switch's argument for 1.0 (memory scarcity) does not apply — 640 GB is abundant. But the quality ceiling of 1.0 (token dropping under routing variance, worse at the small batch-per-expert sizes of a 1B model) is a real cost.
2. **The next increment is not worth it.** ST-MoE measured raising CF 1.25 → 2.0 costs **+7% step time for a 1B-class model and +14% for a 32B model** (TPU, fast all2all), for a quality gain of ~+0.009 that is dominated by the communication cost on *standard* interconnect. On slower interconnect the step-time penalty is strictly worse.
3. **1.25 keeps drops near zero with a well-tuned aux loss** (0.3% in ST-MoE top-1), avoiding the silent information loss of dropped tokens during *pretraining*, where ST-MoE shows dropping does materially hurt (robustness to dropping only holds for fine-tuning).
4. **Eval CF = 2.0** is retained as the inference-time setting (ST-MoE convention) to eliminate drops during evaluation, at no training cost.

---

## 2. Auxiliary-loss coefficient — chosen: **0.01**

**Conflicting positions weighed:**
- **α = 0.01 (dominant default).** Switch Transformer explicitly swept α ∈ [1e-1, 1e-5] in powers of ten and selected **1e-2**: "sufficiently large to ensure load balancing while small enough not to overwhelm the primary cross-entropy objective." Confirmed as the default by ST-MoE, OLMoE (0.01), and NVIDIA Megatron's `moe_aux_loss_coeff`.
- **α = 0.001 (weaker, e.g., Hugging Face's Switch config default).** This is a stability-oriented value; it is too weak to enforce balance under capacity-based dropping, so the capacity factor "does most of the work" via token dropping — exactly the failure mode we are trying to avoid with top-1 routing.
- **α = 0.1 (aggressive).** Forces near-uniform routing, destroying expert specialization (the router "converges to mostly activating the same few experts" is what we want to partially allow; over-regularizing sacrifices task loss). Community guidance places the ceiling around 0.1.
- **α = 0 (DeepSeek-V3-style aux-loss-free balancing).** A principled modern alternative that removes the quality tax of the aux loss using per-expert routing biases. It is *outside the 0.001–0.1 range the problem constrains us to*, so it is recorded here only as the primary revision path (see §4).

**Justification for 0.01:**
1. **It is the empirically swept optimum for the exact mechanism we use.** The Switch sweep is the only published direct search over this range (powers of 10 from 1e-1 to 1e-5), and 1e-2 was the robust interior point.
2. **It is tuned to the joint configuration chosen here.** With top-1 + CF 1.25, balance is what keeps drops low; the design principle is *"the aux loss should be strong enough that the capacity factor rarely needs to drop tokens."* 0.01 achieves the measured 0.3% drop rate (ST-MoE). 0.001 risks routing collapse; 0.1 buys balance by paying a measured modeling-quality tax.
3. **It is consistent across the closest 1B-class precedent.** OLMoE-1B-7B (1B active / 7B total — the same dense-equivalent scale as this budget) trains from scratch with α = 0.01.
4. **Companion term:** add ST-MoE's **router z-loss at 0.001**, which stabilizes training (penalizes large router logits that flip top-1 decisions under bf16 rounding) without a quality trade-off. This is a recommended companion, not one of the three required parameters.

---

## 3. Top-k routing — chosen: **1 (top-1)**

**Conflicting positions weighed:**
- **Top-1 (Switch Transformer).** Fedus et al. (2021) argue k=1 preserves quality while halving router compute, expert capacity, and **communication cost** — all three properties that matter under standard interconnect. Their empirical basis is at 220M FLOP-matched scale, ~50B tokens.
- **Top-2 (GShard, Mixtral, production).** GShard's original motivation (Shazeer's conjecture) is that k>1 gives non-trivial gradients to the router. Mixtral 8×7B uses top-2 successfully, and ST-MoE's larger-scale study (1B FLOP-matched, 100B tokens) finds a small but real top-2 advantage: **+0.004** neg-log-perp over top-1 at equal CF (about 1/20th of a model-tripling), explicitly *revising* Switch's earlier recommendation — while noting the speed difference is negligible **only on TPU-scale fast all2all**.
- **Top-8 fine-grained (OLMoE, DeepSeek-MoE).** The recent 1B-class precedent (OLMoE-1B-7B) actually uses 8-of-64 experts with **dropless** token-choice routing (MegaBlocks-style) — i.e., the modern answer to the capacity-factor problem is to remove static capacity entirely. This is outside the top-1/top-2 framing the problem fixes, and noted as the principled long-term upgrade path.

**Justification for top-1:**
1. **The decision is governed by the interconnect, and top-1 halves the cost.** Each top-2 selection doubles dispatch and combine traffic *and* doubles expert FLOPs for a token. On non-NVLink-full-mesh hardware the all-to-all volume is the binding term; ST-MoE's "slow all2all → smaller CF/n dominates" logic applies to *n* directly.
2. **This is a small (1B) model, which is precisely the regime where Switch found top-1 ≥ top-2.** Switch's own caveat — top-2's small gain only appears at ~5× more training compute (50B → 100B tokens) and ~5× more model FLOPs — means the top-2 premium is unearned at this scale unless we are committed to a very large token budget.
3. **Top-1 removes the double-drop failure mode.** Under capacity-based dropping, a top-2 token whose *both* experts overflow receives zero expert contribution (GShard's overflowed-token case). With CF 1.25 and top-1, a token either gets its expert or a clean residual pass.
4. **Top-1 makes the capacity-factor convention unambiguous** (no k-multiplied capacity), which is operationally valuable given the convention confusion documented above.

**Honest counterweight:** ST-MoE's +0.004 for top-2 is real and might be reproduced; Mixtral proves top-2 is production-viable; and the router-gradient argument for k>1 has not been fully refuted. We accept a bounded quality risk (≤ ~1/20 of a model-tripling) in exchange for a ~2× reduction in the binding resource. This risk is explicitly monitored in the revision triggers below.

---

## Joint consistency check
The three choices are mutually reinforcing rather than independent:
- **top-1 + CF 1.25 + α 0.01** is the exact point Switch Transformer and ST-MoE measured with <0.5% dropped tokens and best Pareto throughput (ST-MoE top-1: CF 1.0→1.25 = +0.011, α 0.01 = 0.3% drops).
- A stronger aux loss (0.1) would be required only if CF were pushed down to 1.0; a weaker one (0.001) would force CF up toward 2.0 to survive imbalance. Keeping CF 1.25 and α 0.01 means neither mechanism has to compensate for the other.
- If we ever move to top-2, the capacity factor must be re-expressed in the Megatron convention (CF ≈ 0.625 to match 1.25·T token-slots), or the config silently doubles both compute and all2all traffic.

---

## 4. Revision conditions

| Trigger (measured, not assumed) | Threshold | Action |
|---|---|---|
| Per-layer dropped-token rate during pretraining | > 1% sustained | Raise train CF to 1.5, then 1.75; re-check step time |
| All-to-all utilization on standard interconnect | > ~70% sustained, or step time regresses >5% vs dense baseline at same quality | Lower CF to 1.0 (accept drops) or switch to top-1-with-dropless kernels (MegaBlocks) |
| Actual interconnect proves fast (e.g., ≥4× measured all2all bandwidth headroom, or upgrade to NVLink full-mesh) | Confirmed by benchmark | Re-test top-2 at matched token-slots (Megatron CF ≈ 0.625) for the +0.004-quality option |
| Eval-gap vs a top-2 reference at matched throughput | > 0.005 neg-log-perp sustained past ~100B tokens | Adopt top-2 with the re-normalized CF |
| Routing collapse observed despite α 0.01 (load-imbalance ratio > 2×) | Per-expert max/mean load | Raise α to 0.03, add router z-loss at 0.001 |
| Aux loss measurably taxing the objective (training loss divergence vs dense of same active size) | > 0.02 | Move to DeepSeek-V3-style aux-loss-free balancing (α = 0, per-expert bias, u = 0.001), keeping CF 1.25 |
| Batch size per expert shrinks substantially (short sequences / small micro-batches) | tokens/E < ~8k | Increase CF toward 1.5 to absorb higher routing variance |

---

## References
- Fedus, Zoph, Shazeer. *Switch Transformers* (2021) — top-1, α-sweep → 0.01, CF 1.0–1.25. https://arxiv.org/abs/2101.03961
- Zoph et al. *ST-MoE* (2022) — CF 1.25/2.0, top-2 recommendation, z-loss 0.001, step-time costs. https://arxiv.org/abs/2202.08906
- Lepikhin et al. *GShard* (2021) — top-2, expert capacity, aux loss. https://arxiv.org/abs/2006.16668
- Zhou et al. *Expert Choice Routing* (2022) — token-choice overprovision 2–8×. https://research.google/blog/mixture-of-experts-with-expert-choice-routing/
- Jiang et al. *Mixtral of Experts* (2024) — top-2, 8 experts, CF 2.0 convention. https://arxiv.org/abs/2401.04088
- Muennighoff et al. *OLMoE* (2024) — 1B active/7B total, dropless top-8/64, α 0.01. https://arxiv.org/abs/2409.02060
- NVIDIA Megatron-Core `moe_utils.py` — `get_capacity`/`switch_load_balancing_loss_func` (top-k convention, α default 0.01). https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_utils.py
- DeepSeek-AI. *Auxiliary-Loss-Free Load Balancing* (2024) — revision path. https://arxiv.org/abs/2408.15664

---

**Bottom line:** top-1 routing, capacity factor **1.25** (train) / 2.0 (eval), auxiliary-loss coefficient **0.01**, with router z-loss 0.001 as a companion. This is the interconnect-bound Pareto point: 2× less dispatch/combine traffic than top-2 for ~1/10th-of-a-tripling in achievable quality, zero meaningful token dropping, and a clearly stated capacity convention that survives scrutiny against the CF 1.0–2.0 / α 0.001–0.1 ranges in the literature.