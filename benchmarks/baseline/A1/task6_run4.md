# MoE Architecture — "Pod-of-4" Cluster on 25 Gbps Ethernet

**Design target:** 1B dense-equivalent parameters · inter-node interconnect capped at 25 Gbps · expert-parallel degree **fixed at 4** (GPUs grouped in pods of 4).

---

## 1. Overview

No published recipe couples low-bandwidth Ethernet with a *fixed, small* expert-parallel degree, so the design centers on one structural decision: **co-locate the expert-parallel group entirely inside one pod of 4 GPUs.** Because EP=4 maps 1:1 onto a pod, every MoE all-to-all exchange stays within the pod, and the expensive 25 Gbps inter-pod links are reserved for data-parallel gradient sync.

The model is a Switch-style (top-1) sparse MoE — not top-2 — because at this bandwidth top-1 halves the token-communication volume (one dispatch + one combine per token instead of two), and with 16 experts the capacity is sufficient at 1B scale. The result is a ~0.97B-parameter footprint with ~84M active parameters per token (~11.6× sparsity).

| Configuration | Value |
|---|---|
| Expert-parallel degree (EP) | **4** (one pod) |
| Data-parallel degree | 16 (16 pods, 64 GPUs total) |
| Per-GPU link (Ethernet) | 25 Gbps (3.125 GB/s) |
| Routing | Top-1 (Switch-style), token-choice, bf16 dispatch |
| Total params | **970,080,256 (~0.97B ≈ 1B dense-equivalent)** |
| Active params per token | ~83.9M |

## 2. Parameters

`d_model = 1,024`, `num_heads = 16` (head_dim 64), `num_layers = 8`, `expert_ffn = 2,048`, `num_experts = 16` per layer, `top_k = 1`, vocab = 128,000, tied embeddings, no bias, RMSNorm.

| Block | Formula | Parameters |
|---|---:|---:|
| Embedding / tied LM head | 128,000 × 1,024 | 131,072,000 |
| Attention (QKV+O) per layer | 4 × 1,024² | 4,194,304 |
| Router per layer | 1,024 × 16 | 16,384 |
| Expert stack per layer | 16 × 3 × 1,024 × 2,048 | 100,663,296 |
| **Per-layer total** | — | **104,876,032** |
| MoE body (8 layers) | 8 × 104,876,032 | 839,008,256 |
| **TOTAL** | — | **970,080,256 (~0.97B)** |
| Active per token | 8×(attn 4.19M + 1 expert 6.29M) | 83,886,080 (~84M) |

- `num_experts = 16` (per layer; 128 expert FFNs total), `top_k = 1`.
- `total_parameters = 970,080,256` → dense-equivalent ≈ 1.0B; active ≈ 0.08B.

## 3. Routing choice with justification

**Top-1 (Switch) gating with a load-balancing auxiliary loss and router z-loss**, trained with deterministic argmax; weights = softmax of `Router(x) = x·W_r` (1,024×16).

Why not top-2 (GShard/Mixtral style):

| Routing | Comm per token (round trip, bf16) | Comm/GPU/step |
|---|---:|---:|
| top_k = 1 | 2 × 2,048 B = 4,096 B | **~403 MB** |
| top_k = 2 | ~4 × 2,048 B ≈ 8,192 B | **~768–806 MB** |

- Top-2 doubles all-to-all traffic (~0.8 GB/GPU/step, ~21 Gbps peak on a single link during the burst) and leaves ~0 headroom against the 25 Gbps cap — unacceptable with fixed EP=4.
- Top-1 at E=16 gives a capacity factor of 16 and halves the per-GPU link load to ~10.7 Gbps peak (>2× margin). At a 1B footprint, top-1 quality loss versus top-2 is small and is the standard trade (Switch Transformer).
- An auxiliary load-balancing loss (coefficient ~0.01) is required to keep routing near-uniform so the 3/4 off-GPU fraction below actually holds; z-loss prevents router logit blow-up.

## 4. Token-communication volume per step and 25 Gbps budget

**Setup:** 64 GPUs = 16 pods × 4; EP=4 inside a pod, DP=16 across pods. Global batch = 1,048,576 tokens/step (16,384 tokens/GPU, e.g., 512 × 2048). bf16 activations (2 B/float).

**Volume per step (top_k=1, per GPU, all 8 expert layers):**

```
Comm = B_gpu × (1 − 1/EP) × 2 × top_k × d_model × 2 B × L_moe
     = 16,384 × (3/4) × 2 × 1 × 1,024 × 2 × 8
     = 402,653,184 B ≈ 403 MB / GPU / step        (per-GPU)
       × 4  = 1.61 GB / pod / step                (intra-pod all-to-all)
       × 16 = ~25.8 GB / step (cluster aggregate)
```

The `(1 − 1/EP) = 3/4` factor is exact: with 16 experts on 4 GPUs each host holds 4 experts, so 3/4 of local tokens route off-GPU. The `2×` is the dispatch (send hidden state) + combine (return expert output) round trip.

**Budget check (worst case: even intra-pod traffic transits a 25 Gbps segment):**
- Sustained: 403 MB / 1 s step ≈ 3.2 Gbps/link (13% of 25 Gbps).
- Peak during a single layer's all-to-all burst (≈37 ms window at a 0.3 s step): 25.2 MB outbound + 25.2 MB inbound per GPU ⇒ **≈10.7 Gbps per 25 Gbps link ≈ 43% utilization**, i.e., >2× headroom including dispatch/combine overlap, retries, and jitter.
- If the pod's intra-node fabric is faster than 25G (NVLink/PCIe), MoE token traffic never touches the 25 Gbps inter-node links at all, consuming **0** of the inter-node budget.

**What actually pressures the 25 Gbps links is DP gradient all-reduce, not MoE tokens:** ~0.97B params × 2 B × 2 × (P−1)/P ≈ **3.6–3.8 GB/link/step** (≈1.2 s of link time at 25 Gbps). Mitigations in §5 keep this inside budget.

## 5. Training implications

- **Load balancing is mandatory.** Top-1 + variable token counts per expert ⇒ auxiliary load-balancing loss per layer (coef ~0.01) + z-loss; monitor expert utilization each step and anneal the aux coef.
- **Overlap and pipeline the all-to-all.** Issue dispatch GEMM, then all-to-all, while the following layer's attention computes; use CUDA-graphs / NCCL-compute overlap. The 10.7 Gbps peak leaves room for this overlap without saturating the link.
- **bf16 (2 B) for every dispatched tensor** — already assumed in the budget; do not use fp32 staging over the network.
- **Hierarchical gradient sync to fit the DP all-reduce:** reduce expert/attention grads inside the pod over the fast local fabric first, then ring all-reduce the per-pod partials across 16 pods on 25 Gbps (≈3.7 GB/link). Optionally 8-bit (BFP/FP8) gradient compression to cut that ~2×.
- **Keep the step compute amortized:** the ~0.16–0.3 s compute step vs ~1.2 s worst-case naive gradient link time means gradient traffic must be compressed/overlapped or the batch lengthened; otherwise training becomes 25 Gbps-bound (target ≤40% of step time on link).
- **EP=4 constrains layer parallelism:** experts per layer must be a multiple of 4 (E=16 ✓); capacity additions come via wider experts (expert_ffn) or more experts, not via more EP ranks.

## 6. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Top-1 quality vs top-2 (higher routing error, more load-imbalance risk) | Medium | Load-balancing + z-loss; fall back to top-2 only if quality gap appears, accepting ~2× comm (~800 MB/GPU/step, still <25 Gbps but tight) |
| Shallow 8-layer depth (vs ~24 for a 1B dense) may underfit | Medium | Monitor eval; deepen via more layers with smaller experts (expert_ffn = 1,536) rather than wider |
| DP gradient all-reduce is the real 25 Gbps consumer (≈1.2 s link time/step) | High | Hierarchical reduce, FP8/8-bit grads, longer steps; without this training is comm-bound (~26% MFU) |
| Expert load spikes under top-1 cause stragglers | Medium | Aux loss + capacity dropout/spillover on hot experts |
| Fixed EP=4 caps max experts per layer and expert batch size | Low | Raise E with d_model/ffn; keep E ≡ 0 (mod 4) |
| Ethernet loss/jitter on all-to-all | Low | bf16 + retransmission budget already within the >2× headroom; prefer flow control / DCB if available |

**Bottom line:** EP=4 co-located in a pod bounds MoE token traffic to **~403 MB/GPU/step (~25.8 GB cluster-wide per step)**, peaking at **≈10.7 Gbps per link — comfortably inside the 25 Gbps cap (>2× margin)**; the residual bandwidth budget goes to a hierarchical, compressed DP gradient all-reduce, which is the design's true network constraint.