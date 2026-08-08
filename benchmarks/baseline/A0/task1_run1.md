# MoE-1B: A 1B Dense-Equivalent Mixture-of-Experts Architecture

## Overview

A decoder-only transformer where each FFN block is replaced by a Mixture-of-Experts (MoE) layer. The model is sized so that **active parameters per token ≈ 1.04B**, matching the compute (FLOPs/token) of a ~1B dense model, while the full sparse checkpoint holds ~5.3B parameters (≈5.1× sparsity). This gives 1B-class serving cost with a larger effective capacity for knowledge.

| Hyperparameter | Value |
|---|---|
| `d_model` | 1,024 |
| `d_ff` (expert MLP width) | 4,096 (4×) |
| Layers `L` | 24 |
| Vocab `V` | 32,000 |
| Experts per layer `E` | 16 |
| Active experts (top-k) | **2** |
| Shared expert per layer | 1 |
| Capacity factor `C` | **1.25** |
| Routing | Learned softmax, top-2 renormalized |
| Auxiliary loss | Load-balancing (Switch-style) + router z-loss |

## Parameters

Expert MLP uses SwiGLU (3 matrices: `W_up`, `W_gate` d→d_ff, `W_down` d_ff→d).

| Component | Math | Params (M) |
|---|---|---|
| Embedding (tied with LM head) | 32,000 × 1,024 | 32.77 |
| Attention per layer | 4 × 1024² | 4.19 |
| Router per layer | 1024 × 16 | 0.016 |
| Per-expert MLP | 3 × 1024 × 4096 | 12.58 |
| Experts per layer | 16 × 12.58 | 201.33 |
| Shared expert per layer | 1 × 12.58 | 12.58 |
| **MoE block per layer** | attn + router + experts + shared | 218.13 |
| **24 layers** | 24 × 218.13 | 5,235.1 |
| **Total** | 32.77 + 5,235.1 | **≈ 5,267.9 M ≈ 5.27B** |

**Active (per token):**

| Component | Math | Params (M) |
|---|---|---|
| Embedding | 32,000 × 1,024 | 32.77 |
| Attention (all layers) | 24 × 4.19 | 100.66 |
| Routed experts (top-2) | 24 × 2 × 12.58 | 603.98 |
| Shared experts | 24 × 12.58 | 301.99 |
| Routers | 24 × 0.016 | 0.39 |
| **Active per token** | sum | **≈ 1,039.8 M ≈ 1.04B** |

FLOPs/token ≈ 6 × active = **~6.2 GFLOPs** — equal to a 1B dense model. Sparsity ratio = 5.27B / 1.04B ≈ **5.1×**.

## Routing Choice

**Learned softmax, top-2, renormalized (Mixtral-style), with a hard shared-expert path.**

- Per token `x`: logits `g = W_router · x` (E logits), softmax over all E, keep top-2, renormalize to probabilities `p_1, p_2`.
- Output `= p_1·E_1(x) + p_2·E_2(x) + Shared(x)`.
- **Why not top-1:** top-1 halves expert compute but measurably hurts downstream quality; single-expert gating is a worse function approximator and less stable during training.
- **Why not top-k>2:** more experts × compute per token, pushes active params past the 1B target with diminishing returns.
- **Why learned (vs. hash/round-robin):** content-based routing is what gives MoE its capacity advantage; deterministic random routing (Switch pre-training experiments) is only useful as a warm-start stage.
- **Why softmax (vs. sigmoid/bias routing):** sigmoid-with-bias (DeepSeek-style) gives finer control and no renormalization drift, but needs extra tuning; top-2 softmax is the proven, well-supported baseline. The shared expert absorbs "generalist" computation and stabilizes the routed experts.

## Training Implications

- **Batch size:** needs large token counts per step so every expert sees enough tokens (ideally ≳ 16×top_k tokens/expert/step) for stable load.
- **Loss:** `L = L_task + α·L_aux + β·L_z`, with `α = 0.01`, `β ≈ 1e-3`:
  - Load balance: `L_aux = E·Σ_e f_e·p̄_e`, where `f_e` = fraction of tokens routed to `e`, `p̄_e` = mean gate prob. Minimized when routing is uniform.
  - Router z-loss stabilizes logits: `L_z = (1/S)·Σ_s (log Σ_e exp(g_e))²`.
- **Capacity handling:** expert capacity per layer = `C · (S·top_k / E)` = 1.25× ideal. Overflow tokens are **dropped** (SendToNextExpert) and diverted to the shared expert; aux loss keeps drop rate <1%.
- **Parallelism:** expert-parallel training with all-to-all token dispatch; routers and shared experts stay replicated.
- **Scale behavior:** MoE reaches a given loss with fewer total tokens but needs longer wall-clock per step; FLOP-efficiency wins at inference, memory (all 5.27B weights resident) is the cost.
- **LR/optimizer:** same schedule as a 1B dense model; keep gradient clipping; warm up the router (or use random routing) for the first ~1k steps.

## Risks

| Risk | Mitigation |
|---|---|
| Routing collapse (all tokens to few experts) | Load-balancing aux loss, z-loss, shared expert, capacity dropping |
| Token dropping degrades quality | C=1.25 buffer, reroute overflow to shared expert, monitor drop rate |
| **Inference memory-bandwidth bound** — all 5.27B params streamed per token even though only 1.04B activate | Bound E (16 keeps expert table small); consider expert offload/quantization |
| Training instability (router oscillation) | z-loss, lower α, longer warm-up, BF16 stability practices |
| Expert redundancy (some experts never specialize) | Aux loss + top-k diversity; periodically check `f_e` |
| Overfitting (5× params, same tokens) | Train with more tokens/epochs or data quality focus |
| All-to-all comms overhead in distributed training | Overlap dispatch with compute; use expert parallelism + sequence packing |

**Bottom line:** 16 experts, top-2, shared expert, capacity 1.25, softmax routing + load-balancing/z-loss — delivers 1B dense-equivalent compute with a 5.3B parameter capacity for ~5.1× parameter-efficiency.