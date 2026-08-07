#!/usr/bin/env python3
"""Establish baseline MoE training throughput metrics from CSV inputs.

Purpose
-------
Given step-time and per-expert token-count CSVs plus a training config
(tokens per step, GPU count, FLOPs/token), this tool computes deterministic
baseline metrics for an MoE run: step-time statistics, tokens/sec (global and
per GPU), a GPU-utilization proxy, expert utilization, a rough MFU estimate,
and a bubble-time proxy. It is an optimization-analysis-phase tool; it
measures the baseline, it does not propose fixes.

Metrics (all deterministic; formulas below)
-------------------------------------------
    tokens_per_sec_global   = tokens_per_step / mean(step seconds)
    tokens_per_sec_per_gpu  = tokens_per_sec_global / gpus

    gpu_util_proxy
        If the steps CSV has a third column ``busy`` (busy seconds within the
        step), the per-step busy fraction is ``busy / seconds`` (clamped to
        [0, 1]) and the proxy is its mean, in percent. This is wall-clock GPU
        utilization.
        Otherwise the proxy is derived from expert counts:
        ``mean(counts) / max(counts) * 100``. This is a PROXY for expert load
        balance, not wall-clock GPU utilization, and is labeled as such.

    bubble_pct              = (1 - mean busy fraction) * 100  (busy data only)

    expert_util_pct         (balanced-capacity definition, matching the
                             moe-debugging analyzer):
        n      = number of experts
        total  = sum(counts)
        balanced = (total / n) * capacity_factor
        expert_util_pct = min(100, total / (n * balanced) * 100)
                       # = 100 / capacity_factor capped at 100
    skew                    = max(counts) / min(counts)   (inf if any is 0)
    top_expert_share_pct    = max(counts) / total * 100

    mfu_pct (ROUGH ESTIMATE)  = tokens_per_sec_global * flops_per_token
                                / (gpus * peak_flops) * 100
        Only computed when --flops-per-token > 0. Not a profiled count: it
        inherits FLOPs/token accuracy from the architecture skill and ignores
        all-to-all and padding overheads.

    Step-time percentiles use the nearest-rank method: for p in (0.5, 0.95),
    index = ceil(p * n) - 1 into the sorted values; median (p50) uses the
    standard two-value average for even n (statistics.median).

Usage
-----
    python3 throughput_profiler.py --steps steps.csv --expert experts.csv \\
        --tokens-per-step 655360 --gpus 8 --flops-per-token 7.6e9 \\
        --capacity-factor 1.25 --output-baseline baseline.csv

Input CSVs
----------
    --steps  header ``step,seconds`` or ``step,seconds,busy`` (seconds per
             training step; ``busy`` = busy seconds within the step).
    --expert header ``expert,count`` or ``expert,count,step`` (per-step or
             cumulative token counts; if a ``step`` column is present, the
             latest step's counts are used).

Output
------
    A baseline CSV (--output-baseline, default ``baseline.csv`` in the current
    directory) with rows ``metric,value,unit``:
    tokens_per_sec_global, tokens_per_sec_per_gpu, p50_step_s, p95_step_s,
    gpu_util_proxy, expert_util_pct, mfu_pct, bubble_pct.
    The same metrics are printed as a table.
"""

import argparse
import csv
import math
import statistics
import sys


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


def non_negative_float(value: str) -> float:
    """argparse type: reject negative floats with a friendly error."""
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a number."
        ) from None
    if parsed < 0.0:
        raise argparse.ArgumentTypeError(
            f"'{value}' must be a non-negative number."
        )
    return parsed


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


def load_steps(path: str) -> tuple[list[float], list[float] | None]:
    """Load a step-time CSV as (seconds, busy_or_None).

    Header ``step,seconds[,busy]``. Returns step seconds and, if the ``busy``
    column is present, the busy seconds per step (aligned by row).
    """
    seconds: list[float] = []
    busy_rows: list[list[str]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise SystemExit(f"error: '{path}' is empty (no header row).")
        columns = [col.strip().lower() for col in header]
        if "step" not in columns or "seconds" not in columns:
            raise SystemExit(
                "error: steps CSV must have columns 'step,seconds[,busy]'; "
                "got: " + ", ".join(header)
            )
        seconds_index = columns.index("seconds")
        busy_index = columns.index("busy") if "busy" in columns else None
        for row in reader:
            if not row or not row[0].strip():
                continue
            seconds.append(float(row[seconds_index]))
            if busy_index is not None:
                busy_rows.append(row)
    if not seconds:
        raise SystemExit(f"error: '{path}' contains no data rows.")
    if busy_index is None:
        return seconds, None
    busy = [float(row[busy_index]) for row in busy_rows]
    return seconds, busy


def load_expert_counts(path: str) -> list[int]:
    """Load an expert token-count CSV as a list of counts.

    Header ``expert,count[,step]``. If a ``step`` column is present, only the
    latest step's counts are used (per-step counts were provided); otherwise
    all rows are used as-is (cumulative or snapshot counts).
    """
    counts: list[int] = []
    step_pairs: list[tuple[int, int]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise SystemExit(f"error: '{path}' is empty (no header row).")
        columns = [col.strip().lower() for col in header]
        if "expert" not in columns or "count" not in columns:
            raise SystemExit(
                "error: expert CSV must have columns 'expert,count[,step]'; "
                "got: " + ", ".join(header)
            )
        count_index = columns.index("count")
        step_index = columns.index("step") if "step" in columns else None
        for row in reader:
            if not row or not row[0].strip():
                continue
            counts.append(int(row[count_index]))
            if step_index is not None:
                step_pairs.append((counts[-1], int(row[step_index])))
    if not counts:
        raise SystemExit(f"error: '{path}' contains no data rows.")
    if step_pairs:
        latest = max(step for _, step in step_pairs)
        counts = [count for count, step in step_pairs if step == latest]
    return counts


def nearest_rank(values: list[float], percentile: float) -> float:
    """Nearest-rank percentile: index = ceil(p * n) - 1 (1-based sort)."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = math.ceil(percentile * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def compute_metrics(
    seconds: list[float],
    busy: list[float] | None,
    counts: list[int],
    tokens_per_step: int,
    gpus: int,
    flops_per_token: float,
    peak_flops: float,
    capacity_factor: float,
) -> dict:
    """Compute all baseline metrics for a run.

    ``counts`` may be empty (no expert CSV provided); busy may be None (no
    busy column). Metrics that need missing data are reported as 0.0 with the
    corresponding *_available flag set to False.
    """
    n_steps = len(seconds)
    step_mean = statistics.fmean(seconds)
    step_p50 = statistics.median(seconds)
    step_p95 = nearest_rank(seconds, 0.95)
    step_min = min(seconds)
    step_max = max(seconds)

    tokens_per_sec_global = tokens_per_step / step_mean
    tokens_per_sec_per_gpu = tokens_per_sec_global / gpus

    # GPU utilization proxy: busy/seconds when busy data present, otherwise an
    # expert-count load-balance proxy (mean/max), clearly not wall-clock util.
    if busy is not None:
        busy_fractions = [
            max(0.0, min(1.0, b / s)) for b, s in zip(busy, seconds)
        ]
        mean_busy_fraction = statistics.fmean(busy_fractions)
        gpu_util_proxy = mean_busy_fraction * 100.0
        gpu_util_source = "busy/seconds"
        bubble_pct = (1.0 - mean_busy_fraction) * 100.0
    else:
        gpu_util_proxy = 0.0
        gpu_util_source = "none"
        bubble_pct = 0.0
        if counts:
            gpu_util_proxy = statistics.fmean(counts) / max(counts) * 100.0
            gpu_util_source = "expert-count proxy (NOT wall-clock GPU util)"

    # Expert utilization (balanced-capacity definition, moe-debugging style).
    n = len(counts)
    total = float(sum(counts))
    if n > 0 and total > 0.0:
        balanced = (total / n) * capacity_factor
        expert_util_pct = min(100.0, total / (n * balanced) * 100.0)
        nonzero = [c for c in counts if c > 0]
        if not nonzero:
            skew = 0.0
        elif min(counts) == 0:
            skew = float("inf")
        else:
            skew = max(counts) / min(counts)
        top_expert_share_pct = max(counts) / total * 100.0
    else:
        expert_util_pct = 0.0
        skew = 0.0
        top_expert_share_pct = 0.0

    # Rough MFU estimate (only when flops/token is supplied).
    if flops_per_token > 0.0:
        mfu_pct = (
            tokens_per_sec_global
            * flops_per_token
            / (gpus * peak_flops)
            * 100.0
        )
    else:
        mfu_pct = 0.0

    return {
        "n_steps": n_steps,
        "step_mean": step_mean,
        "step_p50": step_p50,
        "step_p95": step_p95,
        "step_min": step_min,
        "step_max": step_max,
        "tokens_per_sec_global": tokens_per_sec_global,
        "tokens_per_sec_per_gpu": tokens_per_sec_per_gpu,
        "gpu_util_proxy": gpu_util_proxy,
        "gpu_util_source": gpu_util_source,
        "bubble_pct": bubble_pct,
        "expert_util_pct": expert_util_pct,
        "skew": skew,
        "top_expert_share_pct": top_expert_share_pct,
        "mfu_pct": mfu_pct,
        "expert_data": n > 0,
    }


def baseline_rows(metrics: dict) -> list[tuple[str, float, str]]:
    """Rows for the baseline CSV: (metric, value, unit)."""
    return [
        ("tokens_per_sec_global", metrics["tokens_per_sec_global"], "tokens/s"),
        ("tokens_per_sec_per_gpu", metrics["tokens_per_sec_per_gpu"],
         "tokens/s/gpu"),
        ("p50_step_s", metrics["step_p50"], "s"),
        ("p95_step_s", metrics["step_p95"], "s"),
        ("gpu_util_proxy", metrics["gpu_util_proxy"], "%"),
        ("expert_util_pct", metrics["expert_util_pct"], "%"),
        ("mfu_pct", metrics["mfu_pct"], "%"),
        ("bubble_pct", metrics["bubble_pct"], "%"),
    ]


def write_baseline(metrics: dict, path: str) -> None:
    """Write the baseline CSV (metric,value,unit) to ``path``."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "unit"])
        for metric, value, unit in baseline_rows(metrics):
            writer.writerow([metric, f"{value:g}", unit])


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute baseline MoE training throughput metrics from "
            "step-time and expert-count CSVs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps", required=True,
        help="CSV with header 'step,seconds[,busy]'",
    )
    parser.add_argument(
        "--expert", default=None,
        help="CSV with header 'expert,count[,step]' (per-expert token counts)",
    )
    parser.add_argument(
        "--tokens-per-step", type=positive_int, default=8192,
        help="global tokens per training step",
    )
    parser.add_argument(
        "--gpus", type=positive_int, default=8,
        help="GPU count (for per-GPU throughput and MFU)",
    )
    parser.add_argument(
        "--flops-per-token", type=non_negative_float, default=0.0,
        help="FLOPs/token from the architecture skill; if > 0, compute the "
             "MFU estimate",
    )
    parser.add_argument(
        "--peak-flops", type=positive_float, default=989e12,
        help="per-GPU peak FLOPs (H100 BF16 dense default)",
    )
    parser.add_argument(
        "--capacity-factor", type=positive_float, default=1.0,
        help="expert capacity as a multiple of balanced load (total/n), used "
             "by the expert-utilization definition",
    )
    parser.add_argument(
        "--output-baseline", default="baseline.csv",
        help="path to write the baseline CSV",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()

    seconds, busy = load_steps(args.steps)
    counts = load_expert_counts(args.expert) if args.expert else []

    metrics = compute_metrics(
        seconds=seconds,
        busy=busy,
        counts=counts,
        tokens_per_step=args.tokens_per_step,
        gpus=args.gpus,
        flops_per_token=args.flops_per_token,
        peak_flops=args.peak_flops,
        capacity_factor=args.capacity_factor,
    )

    skew_text = "inf" if math.isinf(metrics["skew"]) else (
        f"{metrics['skew']:.2f}"
    )

    print("MoE throughput baseline")
    print("=======================")
    print(f"  steps                : {metrics['n_steps']}")
    print(f"  step time (s)        : mean {metrics['step_mean']:.3f}  "
          f"p50 {metrics['step_p50']:.3f}  p95 {metrics['step_p95']:.3f}  "
          f"min {metrics['step_min']:.3f}  max {metrics['step_max']:.3f}")
    print(f"  tokens/sec (global)  : {metrics['tokens_per_sec_global']:,.1f}")
    print(f"  tokens/sec per GPU   : {metrics['tokens_per_sec_per_gpu']:,.1f}")

    if metrics["expert_data"]:
        print(f"  expert utilization % : {metrics['expert_util_pct']:.1f}%  "
              f"(capacity factor {args.capacity_factor:g})")
        print(f"  expert skew (max/min): {skew_text}")
        print(f"  top-expert share %   : {metrics['top_expert_share_pct']:.1f}%")
    else:
        print("  expert utilization % : n/a (no --expert CSV provided)")

    print(f"  GPU util proxy %     : {metrics['gpu_util_proxy']:.1f}%  "
          f"({metrics['gpu_util_source']})")
    if metrics["bubble_pct"] > 0.0 or metrics["gpu_util_source"] == "busy/seconds":
        print(f"  bubble time %        : {metrics['bubble_pct']:.1f}%")

    if args.flops_per_token > 0.0:
        print(f"  MFU % (ROUGH EST.)   : {metrics['mfu_pct']:.1f}%")
    else:
        print("  MFU % (ROUGH EST.)   : n/a (pass --flops-per-token)")

    print()
    print("Baseline CSV:")
    print(f"  {args.output_baseline}")
    write_baseline(metrics, args.output_baseline)


if __name__ == "__main__":
    main()
