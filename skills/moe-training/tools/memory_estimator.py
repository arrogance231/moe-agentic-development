#!/usr/bin/env python3
"""Estimate the steady-state per-GPU memory budget for MoE training.

Purpose
-------
Given MoE model parameter counts and a parallelism layout (data, tensor,
pipeline, and expert parallelism), this tool estimates per-GPU training
memory: parameter storage, gradients, optimizer states, activations, and a
rough overhead line. It is a training-setup-phase planning tool; it produces
estimates, not training code.

Formulas (all approximations; marked ROUGH where crude)
-------------------------------------------------------
    dense_params             = total_params - expert_params

    params_per_gpu           = dense_params / (dp * tp * pp) + expert_params / ep
        # Dense parameters are sharded across the full data/tensor/pipeline
        # mesh; expert parameters are sharded across the EP group only.
        # Approximation: DP replicas outside an EP group each hold the full
        # expert set, so the expert term is not divided by DP.

    param_bytes_per_param:     2 (bf16/fp16), 4 (fp32)
    grad_bytes_per_param:      same as param_bytes_per_param
    optim_bytes_per_param:
        adamw + mixed precision: 12  (fp32 master + two fp32 moments)
        adamw + fp32 precision:    8  (two fp32 moments, no separate master)
        sgd   + mixed precision:  8  (fp32 master + momentum)
        sgd   + fp32 precision:   4  (momentum)

    activation_bytes_per_gpu = num_layers * micro_batch_size * seq_len *
                               d_model * 20 / tp *
                               (0.5 if recompute else 1.0)
        # ROUGH: ~20 bytes/token/layer covers attention + FFN intermediates
        # in mixed precision, sharded across tensor parallelism, halved by
        # activation recomputation (gradient checkpointing).

    total_bytes              = (param + grad + optim bytes per param)
                               * params_per_gpu + activation_bytes_per_gpu

    total_with_overhead      = total_bytes * 1.10 + 1.5e9
        # 10% runtime overhead + ~1.5 GB framework/communication buffers.

Usage
-----
    python memory_estimator.py --total-params-b 7.0 --expert-params-b 6.0 \\
        --precision bf16 --optimizer adamw --dp 1 --tp 1 --pp 1 --ep 8 \\
        --gpus 8 --num-experts 64 --micro-batch-size 8 --seq-len 2048 \\
        --d-model 2048 --num-layers 24 --gpu-mem-gb 80
"""

import argparse

GB_BYTES = 1e9


def positive_int(value: str) -> int:
    """argparse type: reject non-positive integers with a friendly error."""
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not an integer."
        ) from None
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            f"'{value}' must be a positive integer."
        )
    return parsed


def positive_float(value: str) -> float:
    """argparse type: reject non-positive floats with a friendly error."""
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a number."
        ) from None
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            f"'{value}' must be a positive number."
        )
    return parsed


def param_bytes_per_param(precision: str) -> int:
    """Storage bytes per parameter for the given precision."""
    if precision in ("bf16", "fp16"):
        return 2
    return 4  # fp32


def optim_bytes_per_param(optimizer: str, precision: str) -> int:
    """Optimizer-state bytes per parameter (approximation).

    Mixed precision keeps an fp32 master copy of every parameter (4 bytes)
    plus one or two fp32 state tensors. Pure fp32 drops the separate master
    (the parameter itself is the fp32 master).
    """
    if optimizer == "adamw":
        return 12 if precision in ("bf16", "fp16") else 8
    return 8 if precision in ("bf16", "fp16") else 4  # sgd


def activation_bytes_per_gpu(
    num_layers: int,
    micro_batch_size: int,
    seq_len: int,
    d_model: int,
    tp: int,
    recompute: bool,
) -> float:
    """Rough steady-state activation memory per GPU (bytes)."""
    multiplier = 0.5 if recompute else 1.0
    return (
        num_layers
        * micro_batch_size
        * seq_len
        * d_model
        * 20.0
        / tp
        * multiplier
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Estimate steady-state per-GPU training memory for an MoE model "
            "under a DP/TP/PP/EP parallelism layout."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--total-params-b", type=positive_float, required=True,
        help="total MoE parameters, in billions",
    )
    parser.add_argument(
        "--expert-params-b", type=positive_float, default=None,
        help="expert-only parameters, in billions "
             "(default: 80%% of --total-params-b)",
    )
    parser.add_argument(
        "--precision", choices=("bf16", "fp16", "fp32"), default="bf16",
        help="training precision",
    )
    parser.add_argument(
        "--optimizer", choices=("adamw", "sgd"), default="adamw",
        help="optimizer used to size state memory",
    )
    parser.add_argument("--dp", type=positive_int, default=1,
                        help="data-parallel degree")
    parser.add_argument("--tp", type=positive_int, default=1,
                        help="tensor-parallel degree")
    parser.add_argument("--pp", type=positive_int, default=1,
                        help="pipeline-parallel degree")
    parser.add_argument("--ep", type=positive_int, default=1,
                        help="expert-parallel degree")
    parser.add_argument(
        "--gpus", type=positive_int, default=None,
        help="GPU count (default: dp*tp*pp*ep); must equal dp*tp*pp*ep "
             "when given",
    )
    parser.add_argument(
        "--num-experts", type=positive_int, default=None,
        help="total experts per MoE layer (optional); --ep must divide it",
    )
    parser.add_argument("--micro-batch-size", type=positive_int, default=8,
                        help="micro-batch size per GPU per step")
    parser.add_argument("--seq-len", type=positive_int, default=2048,
                        help="sequence length")
    parser.add_argument("--d-model", type=positive_int, default=2048,
                        help="hidden/attention dimension")
    parser.add_argument("--num-layers", type=positive_int, default=24,
                        help="number of transformer layers")
    parser.add_argument(
        "--recompute", action="store_true",
        help="enable gradient checkpointing (halves activation memory)",
    )
    parser.add_argument("--gpu-mem-gb", type=positive_float, default=80,
                        help="per-GPU memory limit, in GB")
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()

    if args.expert_params_b is None:
        expert_params_b = args.total_params_b * 0.8
    else:
        expert_params_b = args.expert_params_b
    if expert_params_b >= args.total_params_b:
        parser = build_parser()
        parser.error(
            "--expert-params-b must be less than --total-params-b "
            f"({expert_params_b:g} >= {args.total_params_b:g})."
        )

    # Parallelism consistency checks.
    layout_product = args.dp * args.tp * args.pp * args.ep
    if args.gpus is not None and args.gpus != layout_product:
        parser = build_parser()
        parser.error(
            f"--gpus ({args.gpus}) does not equal dp*tp*pp*ep "
            f"= {args.dp}*{args.tp}*{args.pp}*{args.ep} = {layout_product}."
        )
    if args.num_experts is not None and args.num_experts % args.ep != 0:
        parser = build_parser()
        parser.error(
            f"--ep ({args.ep}) must divide --num-experts "
            f"({args.num_experts}); {args.num_experts} % {args.ep} != 0."
        )
    if args.num_experts is not None and args.num_experts < args.ep:
        parser = build_parser()
        parser.error(
            f"--ep ({args.ep}) cannot exceed --num-experts "
            f"({args.num_experts})."
        )

    gpus = args.gpus if args.gpus is not None else layout_product

    # Model parameter split (dense vs expert).
    total_params = args.total_params_b * GB_BYTES
    expert_params = expert_params_b * GB_BYTES
    dense_params = total_params - expert_params

    # Per-GPU parameter count under the parallelism layout.
    params_per_gpu = (
        dense_params / (args.dp * args.tp * args.pp) + expert_params / args.ep
    )

    param_bpp = param_bytes_per_param(args.precision)
    grad_bpp = param_bpp
    optim_bpp = optim_bytes_per_param(args.optimizer, args.precision)

    params_bytes = params_per_gpu * param_bpp
    grads_bytes = params_per_gpu * grad_bpp
    optim_bytes = params_per_gpu * optim_bpp
    act_bytes = activation_bytes_per_gpu(
        args.num_layers,
        args.micro_batch_size,
        args.seq_len,
        args.d_model,
        args.tp,
        args.recompute,
    )

    total_bytes = params_bytes + grads_bytes + optim_bytes + act_bytes
    total_with_overhead = total_bytes * 1.10 + 1.5e9
    overhead_bytes = total_with_overhead - total_bytes

    limit_bytes = args.gpu_mem_gb * GB_BYTES

    rows = [
        ("Parameters", params_bytes),
        ("Gradients", grads_bytes),
        ("Optimizer states", optim_bytes),
        ("Activations", act_bytes),
        ("Overhead+buffers", overhead_bytes),
        ("TOTAL", total_with_overhead),
    ]

    label_w = max(len(label) for label, _ in rows) + 2
    value_texts = [f"{value / GB_BYTES:.2f} GB" for _, value in rows]
    value_w = max(len(text) for text in value_texts) + 2
    bar = "+" + "-" * (label_w + 2) + "+" + "-" * (value_w + 2) + "+"

    print(
        f"Per-GPU memory budget ({args.precision} precision, "
        f"{args.optimizer} optimizer, DP={args.dp} TP={args.tp} "
        f"PP={args.pp} EP={args.ep}, {gpus} GPU{'s' if gpus != 1 else ''})"
    )
    print(bar)
    print(f"| {'Item':<{label_w}} | {'GB':>{value_w}} |")
    print(bar)
    for (label, _), text in zip(rows, value_texts):
        print(f"| {label:<{label_w}} | {text:>{value_w}} |")
    print(bar)

    print(f"\n  GPU memory limit: {args.gpu_mem_gb:>8.2f} GB")
    if total_with_overhead <= limit_bytes:
        headroom = 100.0 * (limit_bytes - total_with_overhead) / limit_bytes
        free_gb = (limit_bytes - total_with_overhead) / GB_BYTES
        print(f"  Headroom:         {headroom:>8.1f}%  ({free_gb:.2f} GB free)")
    else:
        over_gb = (total_with_overhead - limit_bytes) / GB_BYTES
        print(f"  OVER BUDGET:      total exceeds the GPU limit by "
              f"{over_gb:.2f} GB")


if __name__ == "__main__":
    main()
