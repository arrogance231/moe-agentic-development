# `deepspeed_moe.json` — field-by-field explanation

`deepspeed_moe.json` is strict valid JSON (no comments allowed), so every
explanation lives here. It targets the shared example model: **7B-MoE (~1B
dense-equivalent), 64 experts, top-2, 24 layers, d_model 2048, expert FFN
32768, seq_len 2048**, on **8 x H100-80GB** with the reference layout
`DP=1, TP=1, PP=1, EP=8` (DP is implicit in DeepSpeed configs; see the
batch-geometry note below).

## Batch geometry

| Field | Value | Derivation |
| --- | --- | --- |
| `train_micro_batch_size_per_gpu` | 8 | tokens/expert rule of thumb + activation memory (see worked example) |
| `gradient_accumulation_steps` | 128 | `global / (micro * dp)` = 1024 / (8 * 1) |
| `train_batch_size` | 1024 | `micro * grad_accum * dp` = 8 * 128 * 1 |

DeepSpeed requires `train_batch_size = train_micro_batch_size_per_gpu *
gradient_accumulation_steps * data_parallel_size`. With `EP=8` on all 8 GPUs,
the data-parallel degree is 1 (`DP * TP * PP * EP = 8`), so the accumulation
must do all the lifting to reach the 1024 global batch.

## Optimizer & precision

- **`bf16.enabled: true`** — BF16 mixed precision (preferred over fp16 on
  H100; no loss scaling needed). The AdamW `AdamW` optimizer keeps an fp32
  master copy plus fp32 first/second moments: **12 bytes/param** of optimizer
  state in the memory estimator.
- **`zero_optimization.stage: 2`** — ZeRO-2 shards the optimizer state and
  gradients across the data-parallel group, then all-reduces the reduced
  (partitioned) gradients. With DP=1 the ZeRO benefit is neutral, but the
  block stays for consistency if DP is raised on a bigger cluster.
- **`zero_allow_untested_optimizer` guidance** — do NOT add it. It is only
  needed for optimizers DeepSpeed has not tested; `AdamW` is fully supported.
  If you ever switch to a custom optimizer and DeepSpeed refuses it, the
  escape hatch is `"zero_allow_untested_optimizer": true` — but it must be
  used knowingly, never as a default.
- **`gradient_clipping: 1.0`** — norm clip applied before optimizer step.

## MoE block

| Field | Value | Meaning |
| --- | --- | --- |
| `enabled` | true | turn on DeepSpeed's MoE group/expert-parallel machinery |
| `ep_size` | 8 | expert-parallel degree; 64 experts over 8 ranks = **8 experts per rank** |
| `capacity_factor` | 1.0 | must match the architecture doc; 1.0 drops tokens under routing imbalance (no slack), 1.25 absorbs them |
| `min_capacity` | 4 | minimum tokens an expert is allocated even when under capacity, protecting small expert batches |
| `aux_loss_coef` | 0.01 | load-balancing auxiliary loss weight; in the 0.001–0.01 band from the architecture doc |
| `top_k` | 2 | active experts per token (matches `num_experts_per_tok` in the HF config) |
| `drop_tokens` | false | if true, tokens are dropped rather than routed when an expert is over capacity; false matches `capacity_factor: 1.0` with aux-loss balancing |
| `communication_data_type` | "fp32" | the expert dispatch/combine **all-to-all** runs in fp32 even though parameters are bf16 — keeps the router's weighted sums numerically safe; the price is 2x all-to-all bandwidth vs bf16 |

**All-to-all note.** With `ep_size: 8`, every MoE layer performs a dispatch
all-to-all and a combine all-to-all per micro-batch. On the reference model
that is `8 tokens * 2048 seq * top_k 2 / (64/8) = 4096`
tokens-per-expert per micro-batch per rank per MoE layer (times 24 layers
for the model total) — above the 8–64 utilization floor, which is fine:
the floor is a minimum, not a cap.

## Scheduler

`WarmupCosineLR` with 100 warmup steps over 10,000 total steps, cosine decay
back to `warmup_min_lr` 0. Adjust `total_num_steps` to
`dataset_tokens / global_batch / seq_len` for the real run.

## Why the file has no comments

The JSON spec forbids `//` comments, and DeepSpeed rejects them when parsing.
Every non-obvious field is documented here instead; the JSON itself must stay
parseable by `python -m json.tool`.
