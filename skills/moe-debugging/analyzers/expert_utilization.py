#!/usr/bin/env python3
"""Analyze per-expert token utilization for MoE training.

Purpose
-------
Given a per-expert token-count CSV, this tool computes deterministic
utilization metrics: overall utilization percent, a per-expert utilization
table with at/over-capacity markers, the max/min utilization skew, and a
capacity-overflow estimate. It is a diagnosis-phase tool: it reports how much
of the expert capacity is actually being used, not how to fix it.

Metrics
-------
Let ``n`` be the number of experts, ``total = sum(c_i)`` and
``balanced = (total / n) * capacity_factor``. Then:

    utilization_pct = min(100, total / (n * balanced) * 100)
                    # = 100 / capacity_factor capped at 100
    overflow_experts        = number of experts with c_i > balanced
    overflow_token_fraction = tokens held by over-capacity experts / total
    skew                    = max(c_i) / min(c_i)   (inf if any count is 0)

When ``--total-capacity`` is given it overrides the formula:
``utilization_pct = total / total_capacity * 100``.

Flags
-----
    UNDERUTILIZED  utilization_pct < 70
    OVERFLOW       overflow_token_fraction > 0.1

Usage
-----
    python3 expert_utilization.py --input counts.csv [--capacity-factor 1.0]
                                  [--total-capacity 1000]

Input CSV
---------
A header row ``expert,count`` plus one row per expert.
"""

import argparse
import csv
import math

UNDERUTILIZED = "UNDERUTILIZED"
OVERFLOW = "OVERFLOW"


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


def load_csv(path: str) -> tuple[list[str], list[int]]:
    """Load an expert token-count CSV as (expert_ids, counts)."""
    experts: list[str] = []
    counts: list[int] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise SystemExit(f"error: '{path}' is empty (no header row).")
        columns = [col.strip().lower() for col in header]
        if "expert" not in columns or "count" not in columns:
            raise SystemExit(
                "error: CSV must have columns 'expert,count'; got: "
                + ", ".join(header)
            )
        count_index = columns.index("count")
        for row in reader:
            if not row or not row[0].strip():
                continue
            experts.append(row[0].strip())
            counts.append(int(row[count_index]))
    if not experts:
        raise SystemExit(f"error: '{path}' contains no data rows.")
    return experts, counts


def analyze(
    counts: list[int],
    capacity_factor: float,
    total_capacity: int | None = None,
) -> dict:
    """Compute expert-utilization metrics for a set of expert counts."""
    n = len(counts)
    total = float(sum(counts))
    if total_capacity is not None:
        utilization_pct = total / total_capacity * 100
    elif n > 0 and total > 0.0:
        balanced = (total / n) * capacity_factor
        utilization_pct = min(100.0, total / (n * balanced) * 100)
    else:
        utilization_pct = 0.0

    balanced = (total / n) * capacity_factor if n > 0 else 0.0
    over_experts = [i for i, c in enumerate(counts) if c > balanced]
    overflow_token_fraction = (
        sum(counts[i] for i in over_experts) / total if total > 0.0 else 0.0
    )

    nonzero = [c for c in counts if c > 0]
    if not nonzero:
        skew = 0.0
    elif min(counts) == 0:
        skew = float("inf")
    else:
        skew = max(counts) / min(counts)

    flags: list[str] = []
    if utilization_pct < 70:
        flags.append(UNDERUTILIZED)
    if overflow_token_fraction > 0.1:
        flags.append(OVERFLOW)

    per_expert: list[dict] = []
    for i, count in enumerate(counts):
        share = (count / total) * 100 if total > 0.0 else 0.0
        marker = "OVER" if count > balanced else "OK"
        per_expert.append({
            "expert": i,
            "count": count,
            "share": share,
            "marker": marker,
        })

    return {
        "total": int(total),
        "n": n,
        "utilization_pct": utilization_pct,
        "skew": skew,
        "overflow_experts": len(over_experts),
        "overflow_token_fraction": overflow_token_fraction,
        "flags": flags,
        "per_expert": per_expert,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-expert utilization metrics from an expert "
            "token-count CSV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="CSV with header 'expert,count'")
    parser.add_argument("--capacity-factor", type=positive_float, default=1.0,
                        help="expert capacity as a multiple of the balanced "
                             "load (total/n)")
    parser.add_argument("--total-capacity", type=positive_int, default=None,
                        help="absolute total token capacity; overrides the "
                             "capacity-factor formula")
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    _, counts = load_csv(args.input)
    result = analyze(counts, args.capacity_factor, args.total_capacity)

    skew_text = "inf" if math.isinf(result["skew"]) else (
        f"{result['skew']:.2f}"
    )
    print("Expert utilization analysis")
    print("===========================")
    print(f"  total tokens           : {result['total']}")
    print(f"  experts (n)            : {result['n']}")
    print(f"  utilization %          : {result['utilization_pct']:.1f}%")
    print(f"  skew (max/min)         : {skew_text}")
    print(f"  overflow experts       : {result['overflow_experts']}")
    print(f"  overflow token fraction: {result['overflow_token_fraction']:.4f}")
    flags = result["flags"]
    print("FLAGS: " + (" ".join(flags) if flags else "(none)"))

    print()
    print("Per-expert utilization")
    rows = result["per_expert"]
    header = f"{'expert':>8} | {'count':>8} | {'share%':>7} | {'capacity'}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['expert']:>8} | {row['count']:>8} | "
              f"{row['share']:>6.1f}% | {row['marker']}")


if __name__ == "__main__":
    main()
