#!/usr/bin/env python3
"""Estimate dense-vs-MoE parameter counts and per-token training FLOPs.

Purpose
-------
Given a dense-equivalent model shape (num_layers, d_model, ffn_mult) and
Mixture-of-Experts knobs (num_experts, top_k), this tool prints the dense
baseline parameter count, the full MoE parameter count, the per-token
activated parameters, the MoE/dense parameter ratio, and an approximate
per-token training FLOP count. It is a design-phase planning tool: it
produces numbers, not training code.

Formulas (all counts are parameters unless noted otherwise)
-----------------------------------------------------------
    ffn_dim          = ffn_mult * d_model

    dense_params     = num_layers * (4*d_model**2 + 3*d_model*ffn_dim
                                     + 2*d_model) + vocab*d_model

    moe_params       = num_layers * (4*d_model**2 + 2*d_model
                                     + num_experts * 3*d_model*ffn_dim)
                      + vocab*d_model

    activated_params = num_layers * (4*d_model**2 + 2*d_model
                                     + top_k * 3*d_model*ffn_dim)
                      + vocab*d_model

    flops_per_token  = 6 * activated_params
                      + 4 * num_layers * d_model * seq_len
                      # training, forward + backward

    param_ratio      = moe_params / dense_params

Term-by-term rationale:
    * 4*d_model**2       : attention QKV + output projections (one layer).
    * 3*d_model*ffn_dim  : gated linear unit (GLU) FFN -- two in
                           projections, one out projection.
    * 2*d_model          : layernorm (and bias) parameters, one layer.
    * vocab*d_model      : input (tied) token embedding.
    * Router weights are ~num_experts * d_model per layer and negligible
      at scale, so they are omitted.
    * The 6x factor on activated params is the standard 3x forward +
      3x backward training cost; the additive term covers attention
      complexity over the sequence length.

Usage
-----
    python3 moe_calculator.py --num-layers 24 --d-model 1024 --ffn-mult 4 \
        --num-experts 16 --top-k 2 --vocab 32000 --seq-len 2048

Worked sanity check (the default invocation)
--------------------------------------------
    ffn_dim          = 4 * 1024                             = 4,096
    dense_params     = 24*(4*1024^2 + 3*1024*4096 + 2*1024) + 32000*1024
                     = 435,470,336
    moe_params       = 24*(4*1024^2 + 2*1024 + 16*3*1024*4096) + 32000*1024
                     = 4,965,318,656
    activated_params = 24*(4*1024^2 + 2*1024 + 2*3*1024*4096) + 32000*1024
                     = 737,460,224
    param_ratio      = 4,965,318,656 / 435,470,336         = 11.40
    flops_per_token  = 6*737,460,224 + 4*24*1024*2048      = 4,626,087,936
"""

import argparse
from typing import NamedTuple


class MoEScale(NamedTuple):
    """Computed scale metrics for one dense-vs-MoE comparison."""

    ffn_dim: int
    dense_params: int
    moe_params: int
    activated_params: int
    param_ratio: float
    flops_per_token: int


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


def compute(
    num_layers: int,
    d_model: int,
    ffn_mult: float,
    num_experts: int,
    top_k: int,
    vocab: int,
    seq_len: int,
) -> MoEScale:
    """Compute dense, MoE, and activated scale metrics."""
    ffn_dim = int(ffn_mult * d_model)

    # Per-layer shared blocks: attention projections + layernorms.
    # Router weights (~num_experts * d_model per layer) are negligible.
    attention = 4 * d_model * d_model
    norms = 2 * d_model

    # Per-expert FFN (GLU): two in-projections, one out-projection.
    expert_ffn = 3 * d_model * ffn_dim

    dense_params = (
        num_layers * (attention + expert_ffn + norms) + vocab * d_model
    )
    moe_params = (
        num_layers * (attention + num_experts * expert_ffn + norms)
        + vocab * d_model
    )
    activated_params = (
        num_layers * (attention + top_k * expert_ffn + norms)
        + vocab * d_model
    )

    param_ratio = moe_params / dense_params

    # 6x = 3x forward + 3x backward; attention term adds sequence-length
    # dependence (training, forward + backward).
    flops_per_token = (
        6 * activated_params + 4 * num_layers * d_model * seq_len
    )

    return MoEScale(
        ffn_dim=ffn_dim,
        dense_params=dense_params,
        moe_params=moe_params,
        activated_params=activated_params,
        param_ratio=param_ratio,
        flops_per_token=flops_per_token,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Estimate dense vs MoE parameter counts and per-token "
            "training FLOPs for a given model shape."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-layers", type=positive_int, default=24,
                        help="number of transformer layers")
    parser.add_argument("--d-model", type=positive_int, default=1024,
                        help="hidden/attention dimension")
    parser.add_argument("--ffn-mult", type=positive_float, default=4,
                        help="FFN hidden = ffn_mult * d_model")
    parser.add_argument("--num-experts", type=positive_int, default=16,
                        help="total experts per MoE layer")
    parser.add_argument("--top-k", type=positive_int, default=2,
                        help="active experts per token")
    parser.add_argument("--vocab", type=positive_int, default=32000,
                        help="vocabulary size")
    parser.add_argument("--seq-len", type=positive_int, default=2048,
                        help="sequence length")
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()

    if args.top_k > args.num_experts:
        raise SystemExit(
            f"error: --top-k ({args.top_k}) cannot exceed "
            f"--num-experts ({args.num_experts})."
        )

    m = compute(
        num_layers=args.num_layers,
        d_model=args.d_model,
        ffn_mult=args.ffn_mult,
        num_experts=args.num_experts,
        top_k=args.top_k,
        vocab=args.vocab,
        seq_len=args.seq_len,
    )

    config = [
        ("num_layers", f"{args.num_layers}"),
        ("d_model", f"{args.d_model}"),
        ("ffn_mult", f"{args.ffn_mult:g} (ffn_dim = {m.ffn_dim:,})"),
        ("num_experts", f"{args.num_experts}"),
        ("top_k", f"{args.top_k}"),
        ("vocab", f"{args.vocab}"),
        ("seq_len", f"{args.seq_len}"),
    ]
    print("Estimated MoE architecture")
    print("==========================")
    for label, value in config:
        print(f"  {label:<12}: {value}")

    rows = [
        ("Dense params", f"{m.dense_params:,}"),
        ("MoE params", f"{m.moe_params:,}"),
        ("Activated params", f"{m.activated_params:,}"),
        ("Param ratio (MoE / dense)", f"{m.param_ratio:.2f}x"),
        ("FLOPs per token", f"{m.flops_per_token:,}"),
        ("FLOPs per token (TFLOPs)", f"{m.flops_per_token / 1e12:.1f}"),
    ]

    w1 = max(len(label) for label, _ in rows) + 2
    w2 = max(len(value) for _, value in rows) + 2
    bar = "+" + "-" * (w1 + 2) + "+" + "-" * (w2 + 2) + "+"

    print()
    print("Parameter table")
    print(bar)
    print(f"| {'Metric':<{w1}} | {'Value':>{w2}} |")
    print(bar)
    for label, value in rows:
        print(f"| {label:<{w1}} | {value:>{w2}} |")
    print(bar)


if __name__ == "__main__":
    main()
