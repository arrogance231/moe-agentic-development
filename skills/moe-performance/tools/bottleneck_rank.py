#!/usr/bin/env python3
"""Rank MoE optimization bottlenecks by estimated ROI.

Purpose
-------
Given a candidate list of bottlenecks and, for each, an estimated impact
(fraction of tokens/sec the fix could recover, 0..1), a probability the
bottleneck is actually the limiting factor (0..1), and a cost-to-fix score
(1..10, LOWER = cheaper to fix), this tool ranks candidates by return on
investment and reports an expected-gain band for each. It is an
optimization-analysis-phase tool; it orders the plan, it does not implement
changes.

Formulas (all deterministic)
----------------------------
    roi         = impact * probability / cost
    gain_low    = impact * probability * 0.5
    gain_high   = impact * probability
        # The band is an estimate of the tokens/sec gain fraction (e.g. 0.07
        # means ~7%), not a measurement.

Usage
-----
    python3 bottleneck_rank.py --input candidates.csv --top 5

Input CSV
---------
A header row ``bottleneck,impact,probability,cost`` plus one row per
candidate. ``impact`` and ``probability`` are in [0, 1]; ``cost`` is in
[1, 10] where LOWER means cheaper to fix.

Output
------
    A ranked CSV ``ranked_bottlenecks.csv`` in the current directory, sorted by
    ROI descending. The same ranking is printed as a table.
"""

import argparse
import csv
import json
import sys


def load(path: str) -> list[dict]:
    """Load a candidate CSV as a list of row dicts (numeric fields coerced)."""
    items: list[dict] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise SystemExit(f"error: '{path}' is empty (no header row).")
        columns = [col.strip().lower() for col in header]
        for required in ("bottleneck", "impact", "probability", "cost"):
            if required not in columns:
                raise SystemExit(
                    "error: CSV must have columns "
                    "'bottleneck,impact,probability,cost'; got: "
                    + ", ".join(header)
                )
        name_index = columns.index("bottleneck")
        impact_index = columns.index("impact")
        probability_index = columns.index("probability")
        cost_index = columns.index("cost")
        for row in reader:
            if not row or not row[0].strip():
                continue
            impact = float(row[impact_index])
            probability = float(row[probability_index])
            cost = float(row[cost_index])
            if not (0.0 <= impact <= 1.0):
                raise SystemExit(
                    f"error: impact must be in [0, 1]; got "
                    f"{impact!r} (row '{row[0]}')."
                )
            if not (0.0 <= probability <= 1.0):
                raise SystemExit(
                    f"error: probability must be in [0, 1]; got "
                    f"{probability!r} (row '{row[0]}')."
                )
            if not (1.0 <= cost <= 10.0):
                raise SystemExit(
                    f"error: cost must be in [1, 10]; got {cost!r} "
                    f"(row '{row[0]}')."
                )
            items.append({
                "bottleneck": row[name_index].strip(),
                "impact": impact,
                "probability": probability,
                "cost": cost,
            })
    if not items:
        raise SystemExit(f"error: '{path}' contains no data rows.")
    return items


def expected_gain_band(item: dict) -> tuple[float, float]:
    """Expected tokens/sec gain fraction band (low, high) for an item."""
    base = item["impact"] * item["probability"]
    return base * 0.5, base


def rank(items: list[dict]) -> list[dict]:
    """Return items sorted by ROI descending, each enriched with ROI and the
    expected-gain band (as fractions). The sort is stable on ties."""
    ranked = []
    for item in items:
        low, high = expected_gain_band(item)
        ranked.append({
            **item,
            "roi": item["impact"] * item["probability"] / item["cost"],
            "gain_low": low,
            "gain_high": high,
        })
    ranked.sort(key=lambda row: row["roi"], reverse=True)
    return ranked


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Rank optimization bottlenecks by ROI (impact * probability / "
            "cost) and print expected-gain bands."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="CSV with header 'bottleneck,impact,probability,cost'",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="print only the top N ranked rows",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="print the ranking as JSON instead of a table",
    )
    parser.add_argument(
        "--output", default="ranked_bottlenecks.csv",
        help="path to write the ranked CSV",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    items = load(args.input)
    ranked = rank(items)
    top = ranked[: args.top]

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "rank", "bottleneck", "impact", "probability", "cost", "roi",
            "gain_low", "gain_high",
        ])
        for position, row in enumerate(ranked, start=1):
            writer.writerow([
                position,
                row["bottleneck"],
                f"{row['impact']:g}",
                f"{row['probability']:g}",
                f"{row['cost']:g}",
                f"{row['roi']:.4f}",
                f"{row['gain_low']:.4f}",
                f"{row['gain_high']:.4f}",
            ])

    if args.json:
        payload = {
            "ranked": [
                {
                    "rank": position,
                    "bottleneck": row["bottleneck"],
                    "impact": row["impact"],
                    "probability": row["probability"],
                    "cost": row["cost"],
                    "roi": row["roi"],
                    "gain_band": [row["gain_low"], row["gain_high"]],
                }
                for position, row in enumerate(ranked, start=1)
            ]
        }
        print(json.dumps(payload, indent=2))
        return

    print("Ranked bottlenecks (by ROI = impact * probability / cost)")
    print("==========================================================")
    header = (f"{'rank':>4} | {'bottleneck':<28} | {'impact':>6} | "
              f"{'prob':>5} | {'cost':>4} | {'ROI':>7} | {'gain band':>14}")
    print(header)
    print("-" * len(header))
    for position, row in enumerate(top, start=1):
        band = f"{row['gain_low'] * 100:.1f}-{row['gain_high'] * 100:.1f}%"
        print(f"{position:>4} | {row['bottleneck']:<28} | "
              f"{row['impact']:6.2f} | {row['probability']:5.2f} | "
              f"{row['cost']:4.1f} | {row['roi']:7.4f} | {band:>14}")
    print()
    print(f"Expected gain band = impact * probability (tokens/sec fraction, "
          f"ESTIMATE). Low = half the high.")
    print(f"Ranked CSV: {args.output} ({len(ranked)} rows)")


if __name__ == "__main__":
    main()
