# MoE-1B: A 1B Dense-Equivalent MoE Architecture

## Overview

**"1B dense-equivalent" is defined here as active parameters (and per-token FLOPs) ≈ 1B**, i.e. inference compute matching a ~1B-parameter dense decoder-only transformer. This is the community meaning of the term (cf. Mixtral 8x7B: 47B total / 13B active ≈ "7B-equivalent").

The design is a 12-layer decoder-only transformer with an all-FFN mixture-of-experts: `d_model = 2048`, **8 experts per layer**, **top-2 routing**. This yields **3.49B total parameters and ~1.01B active parameters per token** (3.46:1 sparsity ratio). By construction the per-token FFN compute (805M MACs) exactly matches that of a 1B dense model, which is why it is "dense-equivalent" rather than just "a 3.5B MoE".

| Hyperparameter | Value |
|---|---|
| Layers `N` | 12 |
| `d_model` | 2,048 |
| Attention heads / head_dim | 32 / 64 |
| Expert FFN hidden `d_ff` | 8,192 (4× `d_model`) |
| **num_experts `E`** | **8** (per layer) |
| **top_k** | **2** |
| Vocab (shared in/out embedding) | 32,768 |
| **Capacity factor (CF)** | **1.25** (per-expert cap = ⌈CF·k·T/E⌉) |
| Auxiliary loss | Switch load-balancing, α = 0.01 + router z-loss, β = 1e-4 |
| **Total parameters** | **3,489,955,840 ≈ 3.49B** |
| **Active (dense-equivalent) params** | **1,006,632,960 ≈ 1.01B** |

### Parameter math (explicit)

Per expert FFN (up-proj + down-proj):

```
W_up:   d_model × d_ff      = 2048 × 8192  = 16,777,216
W_down: d_ff   × d_model    = 8192 × 2048  = 16,777,216
per expert                    = 2 × 2048 × 8192     = 33,554,432  (33.6M)
per layer   (E=8)            = 8 × 33,554,432       = 268,435,456 (268.4M)
all experts (N=12)           = 12 × 268,435,456     = 3,221,225,472 (3.22B)
```

Attention (Q,K,V,O), embeddings, router, norm:

```
attention per layer = 4 × d_model²           = 4 × 2048²      = 16,777,216 (16.8M)
  × N=12                                     = 201,326,592     (201.3M)
embeddings          = 32,768 × 2048          = 67,108,864      (67.1M)
router per layer    = 2048 × 8 = 16,384; ×12 = 196,608         (0.2M)
LayerNorms          = 24 × 2048 × 2          = 98,304          (0.1M)

TOTAL = 3,221,225,472 + 201,326,592 + 67,108,864 + 196,608 + 98,304
      = 3,489,955,840 ≈ 3.49B
```

Active params per token:

```
per layer = attention + top-2 experts = 16,777,216 + 2 × 33,554,432 = 83,886,080
× N=12    = 1,006,632,960 ≈ 1.01B   (FFN-only active = 805,306,368 ≈ 805M MACs/token,
                                      equal to a ~1B dense model's FFN compute)
```

## Routing choice

**Strategy: learned softmax top-2.** The router is a single trained linear layer `d_model → E`; logits → softmax over the 8 experts; the top-2 are activated and outputs combined as `Σ p_e · expert_e(x)` re-normalized by `p_1 + p_2`. Routing is **learned** (gradients flow through both router and expert weights), **soft** (probability-weighted, not hard one-hot), and **sparse top-2** (not full soft routing).

**Why top-2 over top-1:** top-1 routing is cheap but under-utilizes experts and degrades into routing collapse (a few experts absorb all traffic) without aggressive balancing. Top-2 doubles representation quality via ensembling, roughly doubles expert utilization, and its 2× FFN cost is still only 25% of a dense FFN — the standard choice at this scale (Mixtral, DeepSeek-V2/V3, Qwen-MoE all use top-2).

**Why not full soft (all experts weighted):** weighting all E experts costs E× FFN compute and destroys the sparsity/FLOPs advantage — that would just be a dense 3.5B model. Top-2 preserves the 1B dense-equivalent compute budget.

**Capacity factor:** per-expert buffer = ⌈CF·k·T/E⌉ = ⌈1.25 × 2T/8⌉ = ⌈0.3125·T⌉ tokens for a batch of T tokens. Note the `k` multiplier: top-2 needs 2T total expert slots, so CF must be scaled by k (CF=1.25 on top of the 2× ideal load). Tokens exceeding capacity are dropped and their loss masked; at CF=1.25 the expected drop rate is <~2%.

**Auxiliary loss:** (1) Switch-style load-balancing `α · E · Σₑ fₑ·Pₑ`, α=0.01, where `fₑ` = fraction of tokens routed to expert e and `Pₑ` = mean router probability — discourages routing collapse. (2) Router z-loss `β/T · Σ log(Σₑ e^{zₑ})²`, β=1e-4 — keeps router logits at stable scale (ST-MoE).

## Training implications

- **Load balance is the dominant concern.** Imbalance → idle expert slots (padded compute) or dropped tokens. The aux loss pulls toward uniform dispatch; the capacity factor trades dropped tokens against buffer padding. α and CF must be tuned jointly.
- **Expert parallelism (EP).** All-to-all token dispatch (2 tokens/expert sent per layer), expert-local matmuls, all-to-all combine. Communication per layer is ~2× hidden dimension per token; with 8 experts and small d_model, keep all-to-all on the critical path cheap via high-bandwidth intra-node links.
- **Batch-size floor.** 8 experts × top-2 means expert batches shrink to ~T/4; throughput efficiency needs large T per step, so bigger batches / longer sequences than a dense 1B model.
- **Training cost vs data.** FLOPs/token ≈ 2 × active ≈ 2 GFLOPs (compute-equivalent to 1B dense), but optimizer memory + gradient sync scale with 3.49B total params → ~10.5B slots (params + Adam's 2× moments) of host memory, with experts sharded across devices.
- **Stability.** Router z-loss + standard gradient clipping; typically keep router learning rate low; mask dropped-token losses to avoid noisy gradients.
- **Data requirement.** Total-parameter/active-parameter ratio 3.5:1 means more total parameters to fit with the same compute budget — expect MoE to need ≥ as much data as the 1B dense baseline for equivalent quality (compute-efficiency over parameter-efficiency).

## Risks

- **Routing collapse / expert starvation:** few experts dominate. Mitigated by top-2, α=0.01 aux loss, and monitoring the min/max per-expert utilization each step.
- **Capacity overflow → dropped tokens:** bursty routing on adversarial or long-tail data can exceed the 0.3125·T buffer; silent token loss degrades quality. Monitor drop rate; raise CF (or drop the k-multiplier, or raise k) if it exceeds ~2%.
- **Communication-bound training:** all-to-all on the critical path can cap utilization below the theoretical 4× FLOPs saving if EP placement is poor.
- **Serving complexity:** 3.49B of weights must live in memory; you only win if inference exploits sparsity (expert offloading, EP, MoE-aware kernels); a naive dense runner pays 3.5× the 1B baseline's bandwidth.
- **Benchmark comparability:** "1B dense-equivalent" must be reported as **active** params; comparing 3.49B total against a dense 1B baseline on latency, throughput, and memory will be misleading without this caveat.
- **Instability from large expert learning rates** and degraded fine-tuning behavior (aux loss must be kept on during fine-tuning to prevent re-collapse).