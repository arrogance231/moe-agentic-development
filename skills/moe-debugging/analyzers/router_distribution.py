#!/usr/bin/env python3
"""Analyze router / expert token distributions for MoE imbalance.

Purpose
-------
Given a per-expert token count (or routing probability) distribution, this
tool computes deterministic imbalance metrics -- normalized entropy, the Gini
coefficient, top-expert share, effective number of experts, utilization skew,
and a capacity-overflow estimate -- and prints diagnostic flags plus an ASCII
histogram. It is a diagnosis-phase tool: it reports what is wrong with the
routing, not how to fix it.

Metrics
-------
Let ``n`` be the number of experts, ``c_i`` the token count (or routing
probability mass) of expert ``i``, ``total = sum(c_i)`` and
``p_i = c_i / total``.

    H_raw     = -sum(p_i * ln(p_i))        # unnormalized entropy
    H_norm    = H_raw / ln(n)              # 0..1 (1 = uniform, 0 = collapse)
    effective = exp(H_raw)                 # effective number of experts, 1..n
    top_share = max(p_i)                   # largest expert's share
    skew      = max(c_i) / mean(c_i)       # utilization skew

    Gini = (2 * sum_i ((i+1) * sorted_c_i)) / (n * total) - (n+1)/n
          # i is the 0-indexed position in the counts sorted ascending, so
          # (i+1) is the 1-indexed rank. This is the standard sample formula;
          # 0 means perfect balance, ~1 means maximal concentration.

    overflow_fraction = (number of experts with
                         c_i > (total/n) * capacity_factor) / n

Entropy interpretation: 1.0 means perfectly uniform routing, 0.0 means all
tokens on a single expert. The effective number of experts ranges from 1
(all mass on one expert) to ``n`` (perfectly balanced).

Flags
-----
    COLLAPSED  top-expert share > 0.5
    IMBALANCED gini > 0.3 or effective experts < 0.5 * n
    OVERFLOW   overflow_fraction > 0.1

Usage
-----
    python3 router_distribution.py --input counts.csv [--capacity-factor 1.0]

Input CSV
---------
A header row plus one row per expert, with either of these column pairs:

    expert,count        # token counts per expert (int)
    expert,probability  # routing probabilities (float)

When the ``probability`` column is present the values are treated as relative
token counts, so every metric above applies unchanged; the printed "total" is
then the sum of the probabilities (~1.0).
"""

import argparse
import math
import sys
from statistics import mean

COLLAPSED = "COLLAPSED"
IMBALANCED = "IMBALANCED"
OVERFLOW = "OVERFLOW"

HISTOGRAM_WIDTH = 40


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


def load_csv(path: str) -> tuple[list[str], list[float]]:
    """Load a router CSV, returning (expert_ids, values).

    ``expert,count`` rows yield integer token counts. ``expert,probability``
    rows yield the routing probabilities as floats; these are treated as
    relative counts by every downstream metric. ``int`` is compatible with
    the declared ``float`` element type, so both cases are covered.
    """
    import csv

    experts: list[str] = []
    values: list[float] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise SystemExit(f"error: '{path}' is empty (no header row).")
        columns = [col.strip().lower() for col in header]
        if "expert" not in columns or not (
            "count" in columns or "probability" in columns
        ):
            raise SystemExit(
                "error: CSV must have columns 'expert,count' or "
                "'expert,probability'; got: " + ", ".join(header)
            )
        use_probability = "probability" in columns
        value_index = columns.index("probability" if use_probability else "count")
        for row in reader:
            if not row or not row[0].strip():
                continue
            experts.append(row[0].strip())
            values.append(float(row[value_index]))
    if not experts:
        raise SystemExit(f"error: '{path}' contains no data rows.")
    return experts, [int(v) if not use_probability else v for v in values]


def entropy(counts: list) -> float:
    """Return normalized entropy H = -sum(p_i ln p_i) / ln(n), in 0..1."""
    total = float(sum(counts))
    n = len(counts)
    if total <= 0.0 or n < 2:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    h_raw = -sum(p * math.log(p) for p in probs)
    return h_raw / math.log(n)


def gini(counts: list) -> float:
    """Return the Gini coefficient over sorted counts (sample formula)."""
    n = len(counts)
    total = float(sum(counts))
    if n == 0 or total <= 0.0:
        return 0.0
    ascending = sorted(counts)
    weighted = sum((i + 1) * value for i, value in enumerate(ascending))
    return (2.0 * weighted) / (n * total) - (n + 1) / n


def analyze(counts: list, capacity_factor: float) -> dict:
    """Compute router-imbalance metrics for a set of expert counts."""
    n = len(counts)
    total = float(sum(counts))
    probs = [c / total for c in counts] if total > 0.0 else [0.0] * n
    h_raw = -sum(p * math.log(p) for p in probs if p > 0)
    h_norm = h_raw / math.log(n) if n >= 2 else 0.0
    effective = math.exp(h_raw) if h_raw > 0.0 else 1.0
    top_share = max(probs) if probs else 0.0
    skew = (max(counts) / mean(counts)) if total > 0.0 else 0.0

    balanced = (total / n) * capacity_factor if n > 0 else 0.0
    overflow = sum(1 for c in counts if c > balanced)
    overflow_fraction = overflow / n if n > 0 else 0.0

    flags: list[str] = []
    if top_share > 0.5:
        flags.append(COLLAPSED)
    if gini(counts) > 0.3 or effective < 0.5 * n:
        flags.append(IMBALANCED)
    if overflow_fraction > 0.1:
        flags.append(OVERFLOW)

    return {
        "total": int(total),
        "n": n,
        "entropy_norm": h_norm,
        "gini": gini(counts),
        "top_share": top_share,
        "effective_experts": effective,
        "skew": skew,
        "overflow_fraction": overflow_fraction,
        "flags": flags,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute MoE router-imbalance metrics (entropy, Gini, effective "
            "experts, skew, capacity overflow) from an expert token-count or "
            "routing-probability CSV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="CSV with header 'expert,count' or "
                             "'expert,probability'")
    parser.add_argument("--capacity-factor", type=positive_float, default=1.0,
                        help="expert capacity as a multiple of the balanced "
                             "load (total/n)")
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    _, counts = load_csv(args.input)
    result = analyze(counts, args.capacity_factor)

    print("Router distribution analysis")
    print("============================")
    print(f"  total tokens         : {result['total']}")
    print(f"  experts (n)          : {result['n']}")
    print(f"  entropy (normalized) : {result['entropy_norm']:.4f}")
    print(f"  gini coefficient     : {result['gini']:.4f}")
    print(f"  top-expert share     : {result['top_share']:.4f}")
    print(f"  effective experts    : {result['effective_experts']:.2f}")
    print(f"  utilization skew     : {result['skew']:.2f}")
    print(f"  overflow fraction    : {result['overflow_fraction']:.4f}")
    flags = result["flags"]
    print("FLAGS: " + (" ".join(flags) if flags else "(none)"))

    print()
    print("Expert distribution (normalized to the busiest expert)")
    total = result["total"]
    top = max(counts)
    for expert, value in zip(range(len(counts)), counts):
        bar_len = round((value / top) * HISTOGRAM_WIDTH) if top > 0 else 0
        bar = "#" * bar_len
        share = (value / total) * 100 if total > 0 else 0.0
        print(f"  e{expert:02d} |{bar:<{HISTOGRAM_WIDTH}}| "
              f"{value} ({share:.1f}%)")


if __name__ == "__main__":
    main()
