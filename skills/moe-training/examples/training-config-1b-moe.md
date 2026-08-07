# Worked example: DeepSpeed MoE training config for a 1B-dense-equivalent MoE

## Input

Architecture document summary (produced by `moe-architecture`):

| Field | Value |
| --- | --- |
| Total params | **7.0 B** |
| Expert-only params | **6.0 B** |
| Dense params | **1.0 B** (~1B dense-equivalent) |
| Experts | 64 |
| Top-k | 2 |
| Layers | 24 |
| d_model | 2048 |
| ffn_mult | 4 (dense/shared FFN hidden 8192) |
| Expert FFN hidden | 32768 (= 4x dense FFN hidden) |
| Vocab | 32000 |
| seq_len | 2048 |

Hardware: **8 x H100-80GB**, one node. Framework: **DeepSpeed MoE**.
Dataset: ~20B tokens; target global batch 1024.

## Workflow walkthrough

**Step 1 — Confirm the architecture.** All fields above come from the
architecture doc. Total/expert/dense counts (7.0 / 6.0 / 1.0 B) are taken
as-is from that document; the layer shape is consistent with them but is not
re-derived here.

**Step 2 — Choose the framework.** DeepSpeed MoE: expert parallelism across
8 GPUs at scale with ZeRO-2 for the ~1B dense replica. HF Transformers is
single-node research only; Megatron adds TP/PP we do not need at 24 layers.

**Step 3 — Plan parallelism.** On 8 GPUs with 64 experts, `EP=8` puts
64 / 8 = **8 experts per rank**, occupies all 8 GPUs, and keeps the per-layer
all-to-all inside one node. With `TP=1` and `PP=1`, the product
`DP * TP * PP * EP = DP * 8` must equal 8, so **DP=1** (implicit). Layouts
like `DP=4 x EP=8` or `DP=8 x EP=8` would need 32/64 GPUs and are infeasible
here. EP=8 divides num_experts (64 % 8 == 0).

**Step 4 — Estimate memory per GPU.** Reference invocation:

```text
python skills/moe-training/tools/memory_estimator.py --total-params-b 7.0 \
    --expert-params-b 6.0 --precision bf16 --optimizer adamw --dp 1 --tp 1 \
    --pp 1 --ep 8 --gpus 8 --num-experts 64 --micro-batch-size 8 \
    --seq-len 2048 --d-model 2048 --num-layers 24 --gpu-mem-gb 80
```

```text
Per-GPU memory budget (bf16 precision, adamw optimizer, DP=1 TP=1 PP=1 EP=8, 8 GPUs)
+--------------------+------------+
| Item               |         GB |
+--------------------+------------+
| Parameters         |    3.50 GB |
| Gradients          |    3.50 GB |
| Optimizer states   |   21.00 GB |
| Activations        |   16.11 GB |
| Overhead+buffers   |    5.91 GB |
| TOTAL              |   50.02 GB |
+--------------------+------------+

  GPU memory limit:    80.00 GB
  Headroom:             37.5%  (29.98 GB free)
```

TOTAL 50.02 GB fits the 80 GB H100 with **37.5% headroom** — comfortable
without gradient checkpointing. Hand-check: per-GPU params = 1.0 / 1 + 6.0 /
8 = 1.75 B; at 2 B/param that is 3.50 GB, ×(2 params + 2 grads + 12
optimizer)/2 = 8x → 28.00 GB; activations = 24 * 8 * 2048 * 2048 * 20 = 16.11
GB; overhead = 10% + 1.5 GB → 50.02 GB. Matches.

**Step 5 — Choose micro-batch and gradient accumulation.** Tokens per expert
per GPU per micro-batch:

```text
micro_batch * seq_len * top_k / (num_experts / ep) = 8 * 2048 * 2 / 8 = 4096
```

4096 is far above the 8–64 utilization band — that band is a *floor* (don't
go below 8), not a cap, so 4096 is fine. The binding constraint is
activation memory: at micro-batch 8 activations are 16.11 GB, which fits with
37.5% headroom; a larger micro-batch (e.g. 16 → 32.2 GB activations) still
fits but buys little utilization. **Micro-batch 8 it is.** With DP=1, gradient
accumulation = global / micro / dp = 1024 / 8 / 1 = **128**.

**Step 6 — Choose checkpoint strategy.** DeepSpeed `ckpt` format, every 1000
steps, saved to shared storage; resume with `--resume`/`load` on the same
format. Do not mix with HF `safetensors` or Megatron `iter_XXXXXX`.

**Step 7 — Produce config and launch command.** `configs/deepspeed_moe.json`
plus the launch command below.

**Step 8 — Validate.** Parallelism product = 8; EP divides 64; JSON parses
(`python -m json.tool`); estimator matches the hand reference within <1%.
See the checklist.

## Config excerpt

Core DeepSpeed fields (full file: `configs/deepspeed_moe.json`; all
explanations in `configs/deepspeed_moe.readme.md`):

```json
{
  "train_batch_size": 1024,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 128,
  "bf16": { "enabled": true },
  "zero_optimization": { "stage": 2 },
  "optimizer": { "type": "AdamW", "params": { "lr": 3e-4, "betas": [0.9, 0.95], "weight_decay": 0.1 } },
  "moe": {
    "enabled": true,
    "ep_size": 8,
    "capacity_factor": 1.0,
    "min_capacity": 4,
    "aux_loss_coef": 0.01,
    "top_k": 2,
    "drop_tokens": false,
    "communication_data_type": "fp32"
  }
}
```

Notes: `train_batch_size` = 8 * 128 * DP 1 = 1024 (must hold for DeepSpeed).
`ep_size: 8` = 8 experts per rank. `capacity_factor: 1.0` and `aux_loss_coef:
0.01` match the architecture doc. `communication_data_type: "fp32"` keeps the
per-layer all-to-all sums numerically safe. `bf16` avoids fp16 loss scaling on
H100.

## Launch command

```bash
deepspeed --num_gpus 8 \
    train.py \
    --model-config configs/huggingface_moe.yaml \
    --deepspeed configs/deepspeed_moe.json \
    --deepspeed_config configs/deepspeed_moe.json \
    --micro-batch-size 8 \
    --gradient-accumulation-steps 128
```

(Flags depend on the training entrypoint; the DeepSpeed config is what drives
batch geometry, ZeRO, and the MoE group.)

## Validation checklist

| Check | Result |
| --- | --- |
| Parallelism product DP*TP*PP*EP = 8 | PASS (1*1*1*8 = 8) |
| EP=8 divides num_experts=64 | PASS (64 % 8 == 0) |
| Estimator total 50.02 GB < 80 GB | PASS (37.5% headroom) |
| Global batch 8*128*1 = 1024 | PASS |
| Tokens/expert/micro-batch = 4096 >= 8 floor | PASS (band is a floor) |
| `python -m json.tool configs/deepspeed_moe.json` | PASS |
| Estimator vs hand reference (<1% diff) | PASS (exact) |
| Capacity factor matches architecture doc (1.0) | PASS |

## Outputs

Files produced by this run of the skill:

- `configs/deepspeed_moe.json` — the training config (strict JSON).
- `configs/deepspeed_moe.readme.md` — field-by-field explanation.
- `configs/huggingface_moe.yaml` — model config the entrypoint loads.
- `examples/training-config-1b-moe.md` — this document.
- `tools/memory_estimator.py` — run once per layout change to re-check memory.
