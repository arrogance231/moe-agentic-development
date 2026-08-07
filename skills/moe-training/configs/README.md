# `configs/` — example MoE training configurations

Three reference configurations for the **same model and hardware**:

| File | Framework | Purpose |
| --- | --- | --- |
| `huggingface_moe.yaml` | HuggingFace Transformers | Mixtral-style model config; single-GPU / small-scale research |
| `deepspeed_moe.json` | DeepSpeed MoE | Expert parallelism at scale with ZeRO-2; `deepspeed_moe.readme.md` explains every field |
| `megatron_moe.sh` | Megatron-LM | Full DP/TP/PP/EP launch script with Megatron's MoE args |

## Shared assumptions

- **Model:** 7B-total-parameter MoE with ~1B dense-equivalent — total params
  **7.0 B**, expert-only **6.0 B**, dense **1.0 B**. The architecture
  document from `moe-architecture` is the source of truth for these counts;
  the layer shape is 24 layers, d_model 2048, ffn_mult 4 (dense/shared FFN
  hidden 8192), **expert FFN hidden 32768 (= 4x the dense FFN hidden)**,
  vocab 32000, seq_len 2048.
- **Routing:** 64 experts, top-2, capacity factor 1.0, aux loss 0.01.
- **Hardware:** 8 x H100-80GB, one node.
- **Parallelism:** DP=1, TP=1, PP=1, EP=8 — 64 experts / 8 ranks = 8 experts
  per rank; DP is implicit. (DP=4 x EP=8 / DP=8 x EP=8 are infeasible on 8
  GPUs: the products would be 32/64.)
- **Batch geometry:** micro-batch 8, gradient accumulation 128, global batch
  1024 (8 * 128 * DP 1).

## Per-file notes

- **HF:** MoE knobs are `num_local_experts`, `num_experts_per_tok`,
  `router_aux_loss_coef`, `capacity_factor`, `router_jitter_noise`.
- **DeepSpeed:** strict valid JSON (no comments); all explanations live in
  `deepspeed_moe.readme.md`. Validate with `python -m json.tool`.
- **Megatron:** `--expert-model-parallel-size 8` with `--num-experts 64`; the
  header comment block documents the layout and the all-to-all per MoE layer.

All three are reference templates for the training-setup phase, not scripts
that are run as-is: substitute real data/checkpoint/tokenizer paths and
adjust scheduler steps to the dataset size.
