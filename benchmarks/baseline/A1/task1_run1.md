# 1B Dense-Equivalent MoE — Architecture Design Document

## 1. Overview

We design a **≈1.0B total-parameter sparse MoE** intended as a **"1B dense-equivalent"** model: it matches the storage/serving footprint of a ~1B-parameter dense LLM while activating only **~0.48B parameters per token**, roughly halving inference/training FLOPs versus a same-size dense baseline. All 14 FFN blocks are MoE (attention stays dense), giving 8 expert FFNs per layer of which 2 are active per token.

**Interpretation note.** "Dense-equivalent" is taken as *total parameter footprint ≈ 1B* (the common model-card reading). Per-token compute is lower (~0.48B active ⇒ ~0.95 GFLOP/token, ≈ half a 1B dense model at equal `d_model`/depth), which is the whole point of the design: keep a 1B-model memory budget, spend ~half the FLOPs, and land quality between a 1B dense and a 0.5B dense model at similar wall-clock cost.

| Config knob | Value |
|---|---|
| `d_model` | 2,048 (16 heads × 128) |
| `num_layers` | 14 (all-MoE FFN) |
| `vocab_size` | 32,000 (tied in/out embeddings) |
| Expert FFN | SwiGLU, intermediate 1,024 |
| **num_experts (E)** | **8** |
| **top_k (K)** | **2** |
| **Routing** | learned, soft (top-2 of softmax), token-choice |
| **Capacity factor** | **1.25** |
| **Auxiliary loss** | Switch/GShard load-balance, α = **0.01** (opt. router z-loss 1e-3) |

## 2. Parameters

### 2.1 Headline figures

| Figure | Count | Digit |
|---|---|---|
| **Total parameters** | **1,005,289,472** | **≈ 1.005B** |
| Active per token | 476,577,792 | ≈ 0.477B |
| **num_experts** | 8 | — |
| **top_k** | 2 | — |
| Expert params (each) | 6,291,456 | ≈ 6.29M |
| FLOPs / token (est.) | ≈ 0.95 GFLOP | ≈ ½ of dense-1B |

### 2.2 Explicit math

Constants: `d² = 2,048² = 4,194,304`.

```
Embedding (tied in/out):
  32,000 × 2,048                     =   65,536,000   (65.54M)

Attention / layer (Wq, Wk, Wv, Wo):
  4 × 2,048²                          =   16,777,216   (16.78M)

Router / layer (linear d_model → E):
  2,048 × 8                           =       16,384   (0.02M)

One expert (SwiGLU: gate+up+down = 3 mats):
  3 × 2,048 × 1,024                   =    6,291,456   (6.29M)

Experts / layer:
  8 × 6,291,456                       =   50,331,648   (50.33M)

MoE layer total:
  16,777,216 + 16,384 + 50,331,648    =   67,125,248   (67.13M)

14 layers:
  14 × 67,125,248                     =  939,753,472   (939.75M)

TOTAL = 939,753,472 + 65,536,000      = 1,005,289,472  ≈ 1.005B

ACTIVE / token:
  embedding           65,536,000
  14 × (attn 16,777,216 + 2 experts × 6,291,456)
                       = 14 × 29,360,128 = 411,041,792
  ACTIVE = 476,577,792 ≈ 0.477B
```

### 2.3 Full parameter table

| Component | Count | Params each | Total | Active / token |
|---|---|---|---|---|
| Embedding (tied) | 1 | 65,536,000 | 65,536,000 | 65,536,000 |
| Attention / layer | 14 | 16,777,216 | 234,881,024 | 16,777,216 |
| Router / layer | 14 | 16,384 | 229,376 | ≈ 16,384 |
| Experts / layer | 14 | 50,331,648 | 704,643,072 | 2 × 6,291,456 = 12,582,912 |
| **Total** | — | — | **1,005,289,472** | **476,577,792** |

## 3. Routing Choice

**Decision: learned, soft, token-choice, top-2 of 8.** Not top-1, not hard/argmax, not fixed/hashed.

**Why top-2 (not top-1).** Top-1 (Switch) halves expert capacity and gives sparser gradient flow, which slows optimization and over-trains a few experts. Top-2 doubles per-token capacity (2×6.29M = 12.58M active FFN per layer), smooths gradients across two experts, and is the best-documented sweet spot (GShard, Mixtral). Top-4+ improves utilization slightly but multiplies all-to-all traffic and imbalance for marginal quality gain at 1B scale.

**Why learned (not hashed/fixed).** Learned routing lets experts specialize (syntax, math, factual clusters); fixed hashing cannot specialize and forfeits the parameter-efficiency argument.

**Why soft (not hard).** The router emits softmax probabilities over all 8 experts; we take the top-2 by score and renormalize. Forward: `output = Σ_{k∈top2} ŵ_k · E_k(x)`, `ŵ_k = softmax(logits)_k / Σ_{j∈top2} softmax(logits)_j`. Gradients flow only to the 2 selected experts (token-choice soft routing), keeping the discrete pick trainable without noisy-top-k sampling at this scale.

**Capacity factor (CF) = 1.25.** Per-expert token budget = `CF × (T_tokens / E)` per layer = `1.25 × 2T/8` under top-2 load. The 25% headroom absorbs router jitter; overflowed tokens are **dropped** (zeroed) during training — acceptable at this scale given the aux loss keeps load near-uniform. CF=1.0 is too brittle; CF≥1.5 wastes memory on idle expert slots.

**Auxiliary load-balancing loss = Switch/GShard, α = 0.01:**
`L_aux = α × E × Σ_e (f_e · P_e)`, where `f_e` = fraction of tokens dispatched to expert `e` and `P_e` = mean router probability for `e` (computed over the top-2 picks). Added to the LM loss **training-only**, never used at inference. Optionally add DeepSeek-V2 router z-loss `1e-3 × mean(logsumexp(router_logits)²)` to stabilize router scale.

## 4. Training Implications

- **Compute.** ~0.95 GFLOP/token ⇒ at fixed batch, roughly **2× throughput vs a 1B dense** baseline and ~30-40% fewer optimizer/GEMV FLOPs, but MoE pays higher memory-bandwidth and communication cost per FLOP, so the practical speedup is ~1.5-2× on ≥4 GPUs.
- **Memory.** 1.005B params ≈ 4.0 GB in fp32 / 2.0 GB in bf16 with AdamW states + gradients ⇒ needs multi-GPU. Experts dominate (704M of 1,005M), so use **expert parallelism (EP)**: shard the 8 experts per layer across devices; keep attention tensor-parallel/dense (Mixtral-style hybrid EP+TP).
- **Communication.** Token-choice top-2 requires an all-to-all dispatch/combine per MoE layer per microbatch; budget ~15-25% of step time and overlap it with attention compute.
- **Optimization.** Standard AdamW, bf16 mixed precision, no dropout inside experts, small-constant router init, and LR warmup identical to a dense 1B model. Effective batch ≥ 512 tokens/layer/step helps router statistics stabilize.
- **Balancing discipline.** Watch `f_e` per layer; expect near-uniform distribution if α=0.01 holds. Since overflow drops tokens, keep CF=1.25 and raise to 1.5 only if p99 expert load saturates capacity.
- **Convergence.** Expect MoE to reach a given loss in fewer tokens than dense (parameter efficiency); quality should land **above a 0.5B dense and near/above a 1B dense at equal total params**, measured with standard perplexity + downstream evals.

## 5. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Router collapse / dead experts** (early or on LR spikes) | High | α=0.01 aux loss, small router init, monitor `f_e`, LR warmup, load-balance loss annealed only after convergence |
| **Capacity overflow / dropped tokens** under tail imbalance | Medium | CF=1.25 headroom, drop-tokens (acceptable at 1B), raise CF or switch to expert-choice routing if p99 saturates |
| **Expert over-specialization / forgetting** during continued training | Medium | Regularized aux loss, periodic expert utilization audit; keep experts small (6.3M) so reuse is cheap |
| **Communication bottleneck (all-to-all)** at scale | Medium | EP sharding, overlap dispatch with attention, reduce K or increase E only with measured comm/comp ratio |
| **Fine-tuning instability** (MoE is more fragile than dense to SFT/LoRA) | Medium | Prefer LoRA/QLoRA on experts, low LR, freeze router or keep aux loss during tuning |
| **Depth at d_model=2,048** (14 layers is shallower than a typical 1B dense) | Low | If quality lags, add 2 layers (raises total ~9% to ~1.09B) rather than widening experts |
| **Data-starved regimes** (route learning overfits small corpora) | Low | Longer warmup, higher dropout only in attention; scale α up early |

**Bottom line:** 1.005B total params (1B dense-equivalent footprint), 8 experts, top-2, learned soft token-choice routing, CF=1.25, aux loss α=0.01 — a conservative, well-trodden recipe that trades ~½ compute per token for expert capacity with bounded routing risk.