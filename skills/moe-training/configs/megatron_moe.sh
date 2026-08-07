#!/usr/bin/env bash
# Megatron-LM launch script for the reference MoE model.
#
# Model:   7B-total MoE (~1B dense-equivalent), 64 experts, top-2,
#          24 layers, d_model 2048, dense FFN 8192, expert FFN 32768,
#          vocab 32000, seq_len 2048.
# Hardware: 8 x H100-80GB.
#
# Parallelism layout
# ------------------
#   TP=1 PP=1 EP=8  ->  DP=1 is implicit (DP = GPUs / (TP*PP*EP) = 1).
#   DP * TP * PP * EP = 8, so all 8 GPUs form one expert-parallel group.
#   64 experts / 8 ranks = 8 experts per rank.
#   Every MoE layer does a dispatch all-to-all + combine all-to-all across
#   the 8 ranks; keep the group inside one node to keep that cheap.
#   Layouts like DP=4 x EP=8 or DP=8 x EP=8 are infeasible on 8 GPUs
#   (product 32/64, not 8) and are NOT used here.
#
# Batch geometry: micro-batch 8, grad-accum 128, global batch 1024.

GPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes 1 \
    --master_addr localhost --master_port 6000"

python -m torch.distributed.run $DISTRIBUTED_ARGS \
    pretrain_gpt.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 8 \
    --num-layers 24 \
    --hidden-size 2048 \
    --ffn-hidden-size 8192 \
    --moe-ffn-hidden-size 32768 \
    --num-attention-heads 32 \
    --num-key-value-heads 8 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --vocab-size 32000 \
    --num-experts 64 \
    --moe-router-topk 2 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.01 \
    --bf16 \
    --sequence-parallel \
    --use-checkpointing-activations \
    --micro-batch-size 8 \
    --gradient-accumulation-steps 128 \
    --global-batch-size 1024 \
    --data-path /path/to/megatron-gpt2-corpus \
    --data-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --split 949,50,1 \
    --train-iters 10000 \
    --lr 3e-4 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --min-lr 0.0 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --save-interval 1000 \
    --eval-interval 500 \
    --eval-iters 100 \
    --save /path/to/checkpoints \
    --load /path/to/checkpoints
