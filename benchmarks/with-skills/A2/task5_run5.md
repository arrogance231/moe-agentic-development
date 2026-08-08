I've gathered the key literature positions. Now let me produce the decision document. Note: I don't have file-write tools in this session, so the document is delivered inline below.

---

# Decision Document — MoE Routing Hyperparameters
**Budget:** 8× 80GB HBM GPUs, standard (non-full-mesh) interconnect | **Target:** 1B dense-equivalent MoE

## Decision summary

| Parameter | Value chosen | Rationale in one line |
| --- | --- | --- |
| Top-k routing | **2** | Training-quality focus; the production default (GShard, Mixtral, Megatron); the aux-loss + capacity-factor choices below control the resulting dispatch cost. |
| Capacity factor | **1.25** | Absorbs routing imbalance (no token loss) with only ~25% padding overhead — the low end of the literature range that the interconnect can afford. |
| Auxiliary-loss coefficient | **0.01** | Strong enough to keep the router balanced at 1.25 CF and 64 experts, while staying at the top of the skill's recommended 0.001–0.01 band (Switch validated 1e-2). |

**Working architecture assumptions** (consistent with the `moe-training` 1B-equivalent example): ~16 layers, `d_model` ~2048, 64 experts, vocab ~32k, seq_len 2048, expert-parallelism `EP=8` (each rank hosts 8 of 64 experts). The hyperparameter logic is robust to small variations in these figures.

---

## 1. Top-k routing — choose **2**

### Positions weighed

- **Top-2 (GShard, Mixtral, Megatron-LM):** The standard for quality-focused training. Mixtral uses `K=2` with 8 experts; GShard uses top-2 gating with capacity `O(2N/E)`; Megatron's Mixtral example sets `--moe-router-topk 2`. Top-2 blends expert knowledge and materially beats top-1 at equal total experts.
- **Top-1 (Switch Transformer):** Halves dispatch (one expert per token), which Switch shows trains faster *per step* and is simpler; best when throughput/latency dominate. The skill's decision table recommends top-1 only for inference-heavy, latency-bound deployment.
- **Top-1 for inference:** Cheapest active parameters, but at the cost of capacity and quality.

### Justification for 2

This is a **training run** on a **1B-equivalent** model, not a latency-bound serving deployment — the skill's default for training is top-2. The quality ceiling matters more than shaving the dispatch factor. The interconnect concern (top-2 ≈ 2× all-to-all volume vs top-1) is real but bounded here: at 1B scale the all-to-all volume is modest (64 experts, 8 ranks, cf=1.25), and we keep the capacity factor low specifically so the top-2 dispatch stays affordable. Switch's per-step throughput advantage is only worth taking when the objective is wall-clock/FLOP efficiency over quality.

---

## 2. Capacity factor — choose **1.25**

### Positions weighed

- **CF = 1.0–1.25 (Switch Transformer):** Empirically Switch performs *better* at low capacity factors; higher CF wastes compute and communication on padding. The skill's rule of thumb: 1.25 absorbs imbalance without much waste; 1.0 is efficient but **drops tokens** under imbalance.
- **CF = 2.0 (GShard top-2; Expert Choice c=2):** GShard sets expert capacity at `2N/E` for top-2 dispatch — effectively CF≈2 — to guarantee near-zero overflow. GLaM likewise uses large capacity factors, explicitly trading compute for quality.
- **CF ≈ full capacity / no drops (Mixtral):** Mixtral uses block-sparse execution (Megablocks) with no dropped tokens — the "capacity factor" argument disappears but the memory/compute cost is that of near-full capacity.

### Justification for 1.25

The decisive constraints are **(a)** dropped tokens hurt quality, and **(b)** capacity factor directly multiplies all-to-all volume (`top_k × tokens × cf`), which is the main risk on a **standard interconnect**. CF=2.0 (GShard/GLaM style) would double padding and dispatch traffic for a guarantee we don't need once the router is load-balanced — at 1B scale with 64 experts there is no overflow-driven accuracy ceiling like at the 100B+ regime. CF=1.0 is the most communication-frugal but drops tokens under any routing skew, which is unacceptable at 1.25... precisely the collapse/imbalance failure mode `moe-debugging` flags. **1.25 sits at the validated sweet spot**: it retains Switch's low-capacity efficiency finding while providing the ~25% buffer that keeps the dropped-token fraction near zero even if the aux loss (0.01) hasn't fully converged the load balance. It also matches the skill's default for a balanced training run.

---

## 3. Auxiliary-loss coefficient — choose **0.01**

### Positions weighed

- **1e-2 (Switch Transformers, Megatron-LM Mixtral example, NVIDIA NeMo):** Switch explicitly **swept 1e-1 → 1e-5** and found **1e-2** "balanced load quickly without interfering with training loss." Megatron's reference Mixtral script ships `--moe-aux-loss-coeff 1e-2`; NeMo's docs recommend the "1e-2 range."
- **1e-3 (Hugging Face `MixtralConfig` default, `router_aux_loss_coef=0.001`):** The conservative end — minimizes distortion of the routing objective but is the weakest load-balancing pressure.
- **Higher (0.1+):** Switch's own sweep found coefficients at the high end begin to interfere with the primary objective; `moe-architecture` caps the sane band at 0.01.

### Justification for 0.01

Two failure modes bracket this choice: too low → **router collapse / imbalance** (with 64 experts, the routing entropy problem is acute, and the skill's debugging guidance is to *raise* 0.001 → 0.01 to fix collapse); too high → the aux term **distorts the router objective**. Choosing **0.01** takes the strongest value in the skill's 0.001–0.01 band and the exact value Switch validated — deliberately, because our chosen **CF=1.25** (not 2.0) and **top-2** make us *depend on good load balance*: if the router skews, 1.25 CF starts dropping tokens, which Mixtral-style no-drop training never experiences. 0.01 keeps the load tight enough that 1.25 CF rarely drops, without reaching the 0.1 regime where Switch observed objective distortion. The HF 0.001 default is the fallback, not the pick, because it leaves the router under-constrained at 64 experts.

---

## Interaction check — the three choices are mutually consistent

- **Top-2 + CF 1.25:** dispatch volume = `2 × tokens × 1.25` = 2.5× the top-1/CF-1.0 baseline. This is the price of quality; it stays affordable because we rejected CF=2.0 (which would push it to 4×) and because 1B-scale all-to-all is small.
- **CF 1.25 + aux 0.01:** the strong aux coefficient is what makes the modest capacity factor safe — it keeps load balanced so 1.25 almost never drops. This is the reason the aux coefficient is set at the *top* of the recommended band rather than the HF 0.001 default.
- **Top-2 + aux 0.01:** the aux loss defined in Switch/Mixtral is written for top-k routing (`tokens_per_expert × router_prob`), so 0.01 behaves exactly as designed here.

---

## Revision conditions (specific triggers, not vibes)

Revise a choice when a measurable signal crosses a threshold, keeping the other two fixed and re-tuning one variable at a time:

| Signal | Threshold | Revision |
| --- | --- | --- |
| Dropped/overflow tokens | > 1% of routed tokens (from `analyzers/router_distribution.py` `OVERFLOW`) | Raise CF 1.25 → **1.5**; if still dropping with healthy entropy, the aux loss is the failure — see next row |
| Router collapse / effective experts | `effective < 0.5 × 64` at steady state, or entropy declining | Raise aux 0.01 → 0.05–0.1 *and* keep jitter on; do **not** fix by inflating CF |
| Aux loss dominating | aux-loss magnitude > ~10% of CE loss, or eval loss regresses when aux is toggled off | Lower aux 0.01 → **0.001**, and accept a slightly higher CF (1.5) to stay drop-free |
| All-to-all dominates step time | communication share > ~40–50% of step, or step time drops materially with lower CF/top-k in an A/B | First cut CF 1.25 → 1.0; if still bound, drop top-2 → **top-1** (this is the only condition that justifies switching routing) |
| Loss plateau with healthy router | No collapse, no overflow, no comm issue | The problem is not routing; stop tuning these three and diagnose elsewhere |

Monitoring must run `router_distribution.py` + `loss_analyzer.py` every ~1k steps so the triggers above are evaluated on data, matching the skill's evidence-driven workflow.