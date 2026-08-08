I'll explore the environment first to find the skill resources referenced, then research the literature positions before producing the decision document.
The search tool timed out. Let me try the other search tool and explore the filesystem.
Good, I have solid literature data. Let me gather a few more specifics (GShard, GLaM, DeepSeek aux-loss conventions) and check for local skill resources in parallel.
I have solid literature coverage now. Let me check the local filesystem for skill resources that could ground the numbers.
Excellent literature coverage. Let me check the local filesystem for any benchmark harness templates or skill resources, and confirm ST-MoE specifics.
Let me check the local filesystem for the harness environment and any skill resources.
I don't have local file tools in this environment, so the deliverable will be the markdown document itself. Let me confirm one more literature point — ST-MoE's capacity-factor and aux-loss findings — since it's the key counterweight to Switch Transformer.
I have everything needed. The ST-MoE paper is the decisive reference — it's the one study that directly measures the CF/quality/step-time trade on a hardware stack where all-to-all is not free, which is exactly this budget's constraint. Writing the decision document now.
# MoE Hyperparameter Decision Document — Capacity Factor, Aux-Loss Coefficient, Top-k

**Budget:** 8× 80GB GPUs, standard (non-NVLink-full-mesh) interconnect
**Target:** 1B dense-equivalent MoE
**Grounding architecture (for reference):** 16 layers, `d_model`=2048, `ffn_mult`=4 (GLU), vocab=32000, seq_len=2048, 16 experts → ~1.14B dense, ~13.2B total MoE (11.6×), ~1.94B activated (≈1.7× dense FLOPs/token).

---

## Decision summary

| Parameter | Chosen value |
| --- | --- |
| **Capacity factor** | **1.25** |
| **Auxiliary-loss coefficient** | **0.01** |
| **Top-k** | **2** |

One-line rationale: **CF = 1.25** is the ST-MoE-tested Pareto-efficient value for a hardware stack where all-to-all is not free; **aux = 0.01** is the coefficient used by every capacity-limited top-2 system (Switch, GShard, GLaM, ST-MoE), the regime we operate in; **top-2** is the production-default quality choice, whose all-to-all cost is acceptable at this small scale on 8 GPUs.

The three are a **coupled package**, not independent picks: top-2 + CF 1.25 + aux 0.01 is exactly the ST-MoE-L / ST-MoE-32B configuration (64 experts, train CF 1.25, eval CF 2.0, `c_B`=0.01, `c_z`=0.001).

---

## Hardware context that drives everything

- **Memory is non-binding.** ~13.2B total params = ~26GB in BF16; with ZeRO-2 sharding of params+grads+AdamW-mixed optimizer across 8 ranks, the per-GPU footprint is ~25–30GB including activations and buffers — far inside 80GB with ≥20% headroom. CF is therefore chosen for **compute/communication reasons, not memory**.
- **The interconnect is the constraint.** Non-full-mesh means all-to-all (dispatch/receive per MoE layer, and EP-gated all-reduce) is comparatively expensive. Every decision below biases away from **dispatch volume** — which scales as `top_k × capacity_factor × tokens`.
- Small scale: 1B dense-equivalent means the all-to-all per token is only `d_model`-sized vectors across 8 ranks — non-trivial but small in absolute terms, and EP=8 puts 2 of 16 experts per rank.

---

## 1. Capacity factor → **1.25**

### Positions weighed

| Position | Source | Value | Notes |
| --- | --- | --- | --- |
| Low, quality-suffices | Switch Transformer (Fedus et al., 2021) | **1.0–1.25** | "Switch Transformers perform better at lower capacity factors (1.0, 1.25)"; CF 2.0→1.25 *slowed* throughput (840→790 steps). <1% tokens dropped with aux loss on. |
| Mid, Pareto-efficient | **ST-MoE (Zoph et al., 2022)** | **1.25 train / 2.0 eval** | The decisive source: measured +0.011 nlp (top-1) / +0.009 (top-2) going 1.0→1.25→2.0, but CF 2.0 is **+7% (ST-MoE-L) and +14% (ST-MoE-32B) slower per step**. Explicitly chose 1.25 over GShard/GLaM's 2.0 for Pareto efficiency, citing all2all cost. |
| High | GShard (Lepikhin 2020), GLaM (Du 2022) | 2.0 (see note) | GShard's top-2 capacity `2S/E` is quoted as CF 2.0 in one convention, but equals **CF 1.0** under the Switch/ST-MoE per-expert definition (`capacity = (top_k·T/E)·CF`). |
| Dropless | Mixtral (2024), DeepSeek-V3 (2024) | no CF / no drops | Both use load-balance mechanisms strong enough to remove the capacity bottleneck entirely (Megablocks variable capacity; bias-based no-aux balancing). Requires engineering we are not committing to. |

**Convention note:** "capacity factor 1.0 to 2.0" in the literature is partly a *definition* artifact. Under the Switch/ST-MoE formula `capacity = (tokens·top_k/E)·CF`, GShard's top-2 `2S/E` is CF **1.0**, not 2.0. ST-MoE's own attribution of "2.0" to GShard/GLaM uses that looser convention. My CF=1.25 is in the Switch/ST-MoE convention and is directly comparable to ST-MoE's 1.25.

### Justification

1. **1.25 captures a real, measured quality gain over 1.0** (+0.009–0.011 neg-log-perp in ST-MoE, ~1/10th the gain of tripling a dense model — small but free-ish), while absorbing the early-training router imbalance that CF 1.0 would turn into dropped tokens. Early training is exactly when the aux loss hasn't yet forced balance.
2. **2.0 is not worth it here.** It doubles padding/dispatch/einsum volume for another +0.009 nlp at +7–14% step time. On **non-full-mesh** interconnect that step-time penalty is disproportionately communication; ST-MoE's own rule — "if the all2all and/or allreduce communications are slow, smaller capacity factors may dominate" — points at 1.25, not 2.0.
3. **Memory doesn't force a lower CF.** On 80GB GPUs there is no pressure to drop to 1.0 for activation-memory reasons (the reason Switch's "smaller is better" applies at trillion-param scale).
4. **We are not dropless.** Mixtral/DeepSeek-V3 avoid a CF by replacing it with strong balancing machinery; at 1B scale on a standard interconnect the engineering cost of dropless execution outweighs the 25% padding it saves.

---

## 2. Auxiliary-loss coefficient → **0.01**

### Positions weighed

| Position | Source | Value | Notes |
| --- | --- | --- | --- |
| 0.001 | Mixtral / HF `MixtralForCausalLM` default | 0.001 | A **dropless** model (Megablocks) — imbalance causes no token drops, so only weak balancing is needed. Not our regime. |
| 0.01 | Switch Transformer | 0.01 | Swept 10⁻¹→10⁻⁵, found 10⁻² "balanced load quickly without interfering with training loss". |
| 0.01 | GShard / GLaM | 0.01 | GLaM explicitly: "GShard aux loss with a 0.01 coefficient." |
| 0.01 | **ST-MoE** | c_B = **0.01** | Same capacity-limited top-2 setup as ours; pairs it with router z-loss c_z=0.001 for logit stability. |
| ~0 | DeepSeek-V3 | α=0.0001 | Only a residual sequence-level guardrail; the *primary* balancer is a bias-based aux-loss-free loop (γ=0.001). Different mechanism, not applicable unless we adopt it. |

The literature disagreement (0.001 vs 0.01 vs "near zero") is really a **routing-regime** disagreement: dropless systems (Mixtral) and bias-based systems (DeepSeek-V3) legitimately run tiny coefficients; every **capacity-limited, token-choice** system converges on 0.01.

### Justification

We run **capacity-limited routing with token dropping enabled** (CF 1.25 is only a buffer, not a guarantee). Under imbalance, tokens get dropped — the one thing the aux loss exists to prevent here. The 0.01 value is therefore the correct end of the spectrum:
- Large enough to keep routing balanced (Switch measured 10⁻² balances "quickly") so CF 1.25's buffer is rarely hit, which keeps drops <1%.
- Small enough not to distort the LM objective — the failure DeepSeek-V3 documents for high α (their Figure shows α=0.01 yielding visibly worse perplexity on a *specific* no-aux experiment; at our scale and with capacity pressure, the balancing benefit outweighs that distortion, and it is the value all comparable systems used).
- 0.001 is a Mixtral value we should **not** copy: without dropless execution, 0.001 invites drift toward imbalance and silent token loss that Mixtral is structurally immune to.

**Recommended pairing:** add ST-MoE's router **z-loss at 0.001** on top of the 0.01 load-balancing term. It penalizes router-logit magnitude and is the documented fix for the small-scale instability that Switch hit before BF16 + careful warmup — a cheap insurance policy at this budget.

---

## 3. Top-k → **2**

### Positions weighed

| Position | Source | Value | Notes |
| --- | --- | --- | --- |
| Top-1 | Switch Transformer (Fedus 2021) | k=1 | Quality-equal to top-2 **at 220M-FLOP scale over 50B tokens**, and faster there because the router was a large compute fraction. |
| Top-1 revised | **ST-MoE** | k=1 vs 2 | At 8× larger scale (100B tokens) top-2 now beats top-1 by **+0.004 nlp** and the speed gap is **"negligible"** — explicitly revising Fedus. |
| Top-2 | GShard, GLaM, Mixtral, ST-MoE-32B, Grok-1, DeepSeek-V2/V3 | k=2 | The production default for training-quality-focused work. |
| Top-k>2 | DeepSeek-V3 | k=8 | 256 experts + group-limited routing — only justified by very high expert counts, not a 16-expert/1B-scale model. |

### Justification

- **Training quality is the objective** (we're choosing training-time hyperparameters: aux loss, CF), and for that, top-2 is the documented sweet spot. ST-MoE's controlled FLOP-matched comparison — the only head-to-head at a comparable regime — shows top-2 winning quality with **negligible** speed cost at this scale, exactly the point where Switch's top-1 advantage evaporates.
- **The all-to-all penalty of top-2 is affordable here.** Top-2 doubles dispatch volume vs top-1, and on non-full-mesh that matters — but our dispatch volume is already kept minimal by CF=1.25, the model is small, and the token vector is only `d_model` wide across 8 ranks. This is the same reasoning ST-MoE used to keep top-2 on their TPU stack.
- **Top-1 is a serving decision, not a training decision.** Revisit it if the benchmark is latency-bound inference; the +0.004 training-quality loss and the loss of expert-blending are the cost.
- **k≥3 is off the table**: with 16 experts, k=3 activates 19% of experts per token, defeating the sparsity that motivates MoE at all.

---

## Coupling check (why the three values cohere)

- **top-2 + CF 1.25**: per-expert capacity = `1.25 × 2T/16` slots; the 25% buffer covers the residual imbalance the aux loss leaves behind, without the 2× padding that top-2 + CF 2.0 would force.
- **CF 1.25 + aux 0.01**: the aux loss keeps utilization high enough that CF 1.25 drops <1% of tokens; a smaller α would make 1.25 drop more; a larger CF would make 0.01's balancing pressure wasteful.
- **top-2 + aux 0.01**: top-2 needs stronger balancing than top-1 (two experts' loads to balance), so the 0.01 end of the scale — not Mixtral's 0.001 — is the coherent pairing.
- This exact triple (k=2, CF=1.25, c_B=0.01) is the configuration ST-MoE shipped for ST-MoE-32B — empirical precedent, not invention.

---

## When I would revise

| Parameter | Revise down when | Revise up when |
| --- | --- | --- |
| **CF → 1.0** | Router entropy stays high and measured dropped-token rate ≈ 0% for ≥ several thousand steps (aux loss is doing the work; saves ~20% dispatch/padding volume). | — |
| **CF → 1.5+** | — | Only if dropped-token rate exceeds ~1% *and* profiling shows communication is *not* the bottleneck (e.g., step-time breakdown shows comm < ~25%). 2.0 never on this interconnect. |
| **Aux → 0.001** | We adopt a dropless implementation (Megablocks/EC-style) or a bias-based balancer; or downstream/val perplexity shows aux-induced distortion at 0.01. | — |
| **Aux → 0.01–0.1** | — | Only if `router_distribution` analysis shows collapse (effective experts < ~0.5×16) or Gini rising despite 0.01. Prefer adding z-loss (0.001) and lowering LR/warmup first. |
| **k → 1** | Inference latency becomes the objective, or profiling shows all-to-all is > ~30% of step time (compare step time at EP=8 vs EP=4). Accept +0.004 nlp loss. | — |
| **k → 3+** | — | Only if expert count rises to ≥ 32–64; never at 16 experts. |

**Trigger for all three:** measured evidence from `throughput_profiler` / `router_distribution` / `loss_analyzer` — dropped-token rate, effective-expert count, router entropy/Gini, and the communication share of step time. No value in this table changes without a number from those diagnostics, and each is validated by a fixed-seed A/B run (per the moe-debugging/moe-performance ablation methodology).