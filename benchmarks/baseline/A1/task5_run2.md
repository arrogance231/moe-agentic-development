# MoE Hyperparameter Decision — 1B-Active Model on 8×80GB

**Decision: top-k = 2, capacity factor (CF) = 1.25, auxiliary-loss coefficient α = 0.01**

## 1. Context and constraints

| Item | Value |
|---|---|
| Hardware | 8 × 80 GB GPUs (≈640 GB HBM), standard (non-full-mesh) interconnect |
| Model | 1B dense-equivalent (≈1B active params/token) |
| Reference architecture | 16 experts (2 per GPU, multiple of 8 devices), all-to-all expert parallelism |

Memory is not binding: a 1B-active / 8B-total model is ~16 GB of BF16 weights (plus optimizer/activations) against 640 GB of HBM. The binding constraints are therefore **compute efficiency** (CF multiplies MoE FLOPs) and **interconnect** (all-to-all volume ∝ tokens × top-k × CF). This is what drives the choices below.

## 2. Top-k routing — choose **k = 2**

**Positions weighed**
- **Top-1 (CF=1.0):** Switch Transformer (Fedus et al., 2021) found top-1 *with* CF=1.0 beat top-2 in their ablations; it is simplest, halves router/all-to-all cost, and pairs naturally with CF=1.0.
- **Top-2 (CF=1.25):** GShard (Lepikhin et al., 2020), ST-MoE (Zoph et al., 2022), and Mixtral 8×7B all standardize on top-2 for better quality-per-FLOP, a soft "ensemble" per token, and resilience to a bad single expert choice.
- DeepSeek-V3's fine-grained routing (top-8 of 256) is out of reach at this scale without shared experts.

**Justification.** At 1B active / 16 experts, top-2 gives materially better quality per unit of active compute (GShard, ST-MoE, Mixtral all confirm), and the 2× all-to-all penalty is **absolute byte volume that is tiny** for an 8B model on an 8-way exchange — a few tens of MB/step, so even a non-mesh interconnect is not saturated. ST-MoE is the closest scale-class precedent and explicitly recommends top-2. The Switch top-1 result was a systems-driven call at much larger scale; its main advantage (less comms) is not decisive here.

## 3. Capacity factor — choose **CF = 1.25**

Capacity per expert = ⌈(T·K/E)·CF⌉.

**Positions weighed**
- **CF=1.0:** Switch's default; minimizes FLOPs and comms but drops a meaningful token fraction at moderate batch sizes, especially with top-2.
- **CF=1.25:** GShard / ST-MoE standard for top-2; caps padding at ~25% over fair share.
- **CF=1.5–2.0:** Used for small batches / fine-tuning; expert-choice routing uses CF≈2.0 to match top-2 compute. Wasteful when the goal is compute- or comm-bound.

**Justification.** With top-2, the second-choice assignment is more skewed than first-choice, so CF=1.0 drops too many tokens (poor quality); CF=2.0 wastes up to ~50%+ of MoE FLOPs in padding and doubles the all-to-all traffic on an interconnect that is explicitly **not** full-mesh. CF=1.25 is the empirically validated sweet spot for top-2 training (GShard, ST-MoE), keeping dropped tokens <1–2% when paired with a working aux loss while holding compute/comms overhead near the low end.

## 4. Auxiliary-loss coefficient — choose **α = 0.01**

L = L_task + α·L_balance (+ optional z-loss at 0.001).

**Positions weighed**
- **α = 0.001:** Near no-balancing; risks expert collapse at 16 experts; only safe where a strong hard cap (low CF) does the work.
- **α = 0.01:** Switch and GShard default; ST-MoE's recommended operating point (with z-loss 0.001).
- **α = 0.1:** Strong balancing, but documented to degrade task quality — the "excessive auxiliary loss" failure case.
- **α = 0 (aux-loss-free):** DeepSeek-V3's approach — but it requires sigmoid routing + a per-expert bias schedule; not available on a plain softmax top-k router.

**Justification.** At 16 experts, α=0.001 leaves the router free to collapse and pushes all balancing onto the hard cap (wasted compute, dropped tokens). α=0.1 measurably sacrifices model quality to achieve cosmetic balance. α=0.01 is the robust middle that keeps per-expert load even enough that CF=1.25 rarely drops tokens, without over-regularizing. It is also the value every widely-deployed top-2 system (GShard, ST-MoE, Megatron/NeMo defaults) was tuned around.

## 5. Monitoring and revision triggers

Monitor: per-expert token fraction, dropped-token fraction, aux-loss magnitude, and all-to-all share of step time.

| Signal | Action |
|---|---|
| Dropped tokens >1–2% | First raise α to 0.02; if still high, raise CF to 1.5 |
| Expert collapse (>90% tokens on a few experts) | Raise α toward 0.05, add z-loss (α_z=0.001), shrink router-init std, or add random-routing warmup |
| All-to-all >~30% of step time | Lower CF to 1.0 and/or drop to top-1 (CF=1.0, α=0.01) |
| MoE quality below dense-1B baseline | Reconsider capacity ratio (more experts) or CF=1.5 |
| Small batches (fine-tuning, short seqs) | Raise CF to 1.5 (small-batch imbalance is intrinsically worse) |
| Future move to full-mesh/NVLink node | CF=1.5 becomes affordable; revisit the CF cap upward |

The top-k=2 choice is the least likely to flip — it is the standard for this scale class — while CF and α should be re-tuned jointly, since they interact: a tight CF lets a weak α pass, and a loose CF demands a stronger α.