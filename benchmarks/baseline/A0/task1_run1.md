# M1B-MoE-16x2 — A 1B-Parameter Dense-Equivalent MoE

*Scope note: "dense-equivalent" here means **total parameter count ≈ 1B** with a 2.9× sparsity factor (active ≈ 0.37B per token). This targets ~1B-dense-class quality at ~1/3 the per-token FLOPs.*

## 1. Overview

A decoder-only Transformer where every FFN block is replaced by a Mixture-of-Experts (MoE) layer. Shared non-MoE weights (embeddings, attention, routers) are always active; per token, only 2 of 16 experts run. All configs use power-of-2 sizes for clean parallelism.

| Config | Value |
|---|---|
| Vocab size | 32,768 |
| d_model / d_head / heads | 2,048 / 128 / 16 |
| Layers | 12 |
| Sequence length | 2,048 |
| **num_experts** | **16** |
| **top_k (active experts)** | **2** |
| Expert intermediate | 1,024 (d_model / 2) |
| **Capacity factor** | **1.25** |
| **Aux loss** | Load-balancing (α=0.01) + router z-loss (β=0.001) |
| Total params | ≈ 1.07B |
| Active params / token | ≈ 0.37B (34% density) |

## 2. Parameters

Explicit math (tied embedding + LM head):

- **Embedding + LM head:** `32,768 × 2,048 = 67,108,864` = 67.1M
- **Attention (per layer):** `Q,K,V + O = 4 × 2,048² = 16,777,216` → ×12 = **201.3M**
- **Expert FFNs (per expert):** `2 × 2,048 × 1,024 = 4,194,304` → ×16 experts = 67.1M/layer → ×12 = **805.3M**
- **Routers:** `12 × (2,048 × 16) = 393,216` = **0.4M**
- **Total:** `67.1 + 201.3 + 805.3 + 0.4 = 1,074.1M ≈ 1.07B`
- **Active per token:** `67.1 (emb) + 201.3 (attn) + 12 × 2 × 4.19M (top-2) + 0.4 (router) = 369.5M ≈ 0.37B`
- **Density:** `369.5 / 1074.1 = 34.4%` → **sparsity factor ≈ 2.9×**

| Component | Formula | Params (M) |
|---|---|---|
| Embedding + LM head (tied) | 32,768 × 2,048 | 67.1 |
| Attention (12 layers) | 12 × 4 × 2,048² | 201.3 |
| MoE expert FFNs (12 × 16) | 12 × 16 × 2 × 2,048 × 1,024 | 805.3 |
| Routers (12 layers) | 12 × 2,048 × 16 | 0.4 |
| **Total** | | **1,074.1 (≈1.07B)** |
| **Active per token** | 67.1 + 201.3 + 100.7 + 0.4 | **369.5 (≈0.37B)** |

## 3. Routing Choice

**Learned softmax, top-2, soft routing (weighted sum).** A linear layer projects `d_model → 16` logits; a softmax yields probabilities; the top-2 indices are selected, probabilities renormalized, and the two expert outputs are combined as `Σ wᵢ · FFNᵢ(x)`.

**Why not top-1 (hard)?** Top-1 (Switch-style) maximizes expert sparsity but is fragile — a single misrouted token gets a single, possibly noisy expert, and there is zero redundancy during training. Top-2 provides two routes per token: better quality with marginal FLOP overhead (Mixtral demonstrates this at 8 experts; we apply it at 16), smoother gradients (both experts receive updates proportional to their weights), and natural load spreading (collisions are cut in half).

**Why learned, not fixed/hash?** Learned routing adapts to the data distribution and is trainable end-to-end with standard backprop; fixed hash routing requires hand-designed partitionings and cannot exploit expert specialization. Soft (weighted) routing — versus hard one-hot selection — lets the renormalized softmax weights act as a differentiable credit allocator, avoiding the high-variance gradients of hard Gumbel-style sampling.

## 4. Training Implications

- **Compute efficiency:** sparse activation gives 2.9× parameter-to-FLOP efficiency; only experts in `top_k` run. Wall-clock gains require expert parallelism — on fewer than 8 GPUs, all-to-all token dispatch (per-MoE-layer) can dominate and erase gains.
- **Memory:** all 16 experts per layer are resident (805.3M of 1.07B params). Load with expert parallelism (EP) across ranks; embed + attention with data/sequence parallelism. Expected tokens/expert at 256×2048 batch: `524,288/16 = 32,768`, buffered to `1.25 × 32,768 = 40,960` slots/layer; overflow tokens fall back through the residual path (kept for training, disabled for eval).
- **Aux losses:** load-balancing loss (α=0.01) penalizes deviation from uniform token assignment to prevent routing collapse; z-loss (β=0.001) penalizes large router-logit magnitudes, stabilizing training (loss-spike reduction). Anneal both coefficients in the final ~10% of training.
- **Initialization:** initialize all experts identically from a trained (or He-uniform) dense FFN checkpoint — Switch-style replication — with the same seed across experts to avoid early divergence.
- **Dropout:** apply only *inside* experts, never on the shared paths or the router (Switch finding: shared-path dropout hurts).
- **Batch-size pressure:** experts need enough tokens/layer to stay populated; small batches starve experts and inflate the aux loss, so run large effective batches (gradient accumulation) at this 1B scale.

## 5. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Routing collapse → dead experts | Quality cliff, wasted capacity | Load-balancing loss + CF headroom |
| Token dropping at capacity | Degradation at long seqs / streaming | CF 1.25, no-drop eval |
| Communication-bound all-to-all | Gains vanish at small batch / few GPUs | EP with ≥8 GPUs, large batches |
| Expert fragmentation | Idiosyncratic specialization limits scaling to ≥64 experts | Keep 16 experts; monitor per-expert utilization |
| Training instability / loss spikes | Divergence in late training | z-loss, aux-coefficient annealing |
| Per-expert overfitting | 1/8 of tokens per expert → higher variance at 1B scale | Weight decay inside experts, dropout |
| Memory bound | 805M of 1.07B params must always be resident | Acceptable at this size; expert-count scaling is memory-bound, not compute-bound |

**Bottom line:** 16 experts / top-2 / learned soft routing / CF 1.25 / load-balance + z-loss is the config that maximizes quality-per-parameter while keeping sparsity high enough (2.9×) to make MoE worthwhile at ~1B total parameters.