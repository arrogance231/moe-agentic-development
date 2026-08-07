#!/usr/bin/env python3
"""Analyze a training loss curve for MoE numerical failure signatures.

Purpose
-------
Given a ``step,loss`` curve, this tool detects deterministic flags for the
failure classes that show up in MoE training loss logs: NaN/Inf values,
loss spikes, loss plateaus, and divergence. It is a diagnosis-phase tool: it
reports what the curve shows, not how to fix it.

Flags
-----
    NAN         number of rows whose loss is NaN
    INF         number of rows whose loss is +-inf
    SPIKE       number of steps where
                |loss[i] - rolling_median(11)| > 5 * MAD(11),
                where the window is the up-to-11 rows centred on step i
                (clipped at the curve ends, only finite rows counted) and
                MAD is the median absolute deviation of that window. The
                flag also carries the top-5 spike steps by deviation.
    PLATEAU     the last 50 rows (or all, if fewer) have a linear-regression
                slope within +-0.001/step AND the final loss is above
                1.5 x the minimum finite loss in the curve.
    DIVERGENCE  the last 20 finite rows are monotonic non-decreasing.
    GOOD        none of the above flags fired.

NaN/Inf values are never treated as spikes; a rolling window needs at least
three finite points and a non-zero MAD before it can flag a spike.

Usage
-----
    python3 loss_analyzer.py --input loss.csv

Input CSV
---------
A header row ``step,loss`` followed by one row per logged step. Missing,
``nan``, ``inf``, ``-inf`` and other non-numeric loss cells are accepted and
converted to NaN.
"""

import argparse
import csv
import math
from statistics import median

SPIKE_WINDOW = 11
SPIKE_TOP_N = 5
PLATEAU_WINDOW = 50
PLATEAU_MAX_SLOPE = 0.001
PLATEAU_RATIO = 1.5
DIVERGENCE_WINDOW = 20


def to_float(text: str) -> float:
    """Convert a CSV cell to a float, returning NaN for non-numeric text."""
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered in ("nan", "none", ""):
        return float("nan")
    if lowered in ("inf", "+inf", "infinity", "+infinity"):
        return float("inf")
    if lowered in ("-inf", "-infinity"):
        return float("-inf")
    try:
        return float(stripped)
    except ValueError:
        return float("nan")


def load_csv(path: str) -> list[tuple[int, float]]:
    """Load a loss curve as (step, loss) pairs; non-numeric loss -> NaN."""
    curve: list[tuple[int, float]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise SystemExit(f"error: '{path}' is empty (no header row).")
        columns = [col.strip().lower() for col in header]
        if "step" not in columns or "loss" not in columns:
            raise SystemExit(
                "error: CSV must have columns 'step,loss'; got: "
                + ", ".join(header)
            )
        loss_index = columns.index("loss")
        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                step = int(row[0].strip())
            except ValueError:
                continue
            curve.append((step, to_float(row[loss_index])))
    if not curve:
        raise SystemExit(f"error: '{path}' contains no data rows.")
    return curve


def linear_regression(points: list[tuple[float, float]]) -> float:
    """Return the slope of the least-squares fit through ``points``."""
    n = len(points)
    if n < 2:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0.0:
        return 0.0
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    return cov / var_x


def detect_spikes(curve: list[tuple[int, float]]) -> tuple[list[int], int]:
    """Return (top spike steps by deviation, total spike count)."""
    losses = [loss for _, loss in curve]
    spikes: list[tuple[float, int]] = []
    for i, loss in enumerate(losses):
        if not math.isfinite(loss):
            continue
        lo = max(0, i - (SPIKE_WINDOW // 2))
        hi = min(len(losses), i + SPIKE_WINDOW // 2 + 1)
        window = [losses[j] for j in range(lo, hi) if math.isfinite(losses[j])]
        if len(window) < 3:
            continue
        rolling_median = median(window)
        mad = median(abs(x - rolling_median) for x in window)
        if mad == 0.0:
            continue
        deviation = abs(loss - rolling_median)
        if deviation > 5.0 * mad:
            spikes.append((deviation, curve[i][0]))
    spikes.sort(reverse=True)
    return [step for _, step in spikes[:SPIKE_TOP_N]], len(spikes)


def detect(curve: list[tuple[int, float]]) -> dict:
    """Detect flags on a loss curve; returns a dict of summary metrics."""
    finite = [(step, loss) for step, loss in curve if math.isfinite(loss)]
    losses = [loss for _, loss in finite]
    min_loss = min(losses) if losses else float("nan")
    max_loss = max(losses) if losses else float("nan")
    final_loss = losses[-1] if losses else float("nan")

    nan_count = sum(1 for _, loss in curve if math.isnan(loss))
    inf_count = sum(1 for _, loss in curve if not math.isfinite(loss)
                    and not math.isnan(loss))

    spike_steps, spike_count = detect_spikes(curve)

    # Plateau: slope of the last PLATEAU_WINDOW finite rows within tolerance
    # AND final loss well above the curve minimum.
    tail = finite[-PLATEAU_WINDOW:]
    plateau = False
    if len(tail) >= 2 and math.isfinite(final_loss) and math.isfinite(min_loss):
        slope = linear_regression(
            [(float(step), loss) for step, loss in tail]
        )
        plateau = (
            abs(slope) <= PLATEAU_MAX_SLOPE
            and final_loss > PLATEAU_RATIO * min_loss
        )

    # Divergence: last DIVERGENCE_WINDOW finite rows monotonic non-decreasing.
    recent = finite[-DIVERGENCE_WINDOW:]
    divergence = False
    if len(recent) >= 2:
        divergence = all(
            recent[i][1] >= recent[i - 1][1] for i in range(1, len(recent))
        )

    flags: list[str] = []
    if nan_count > 0:
        flags.append("NAN")
    if inf_count > 0:
        flags.append("INF")
    if spike_count > 0:
        flags.append("SPIKE")
    if plateau:
        flags.append("PLATEAU")
    if divergence:
        flags.append("DIVERGENCE")
    if not flags:
        flags.append("GOOD")

    return {
        "steps": len(curve),
        "min_loss": min_loss,
        "max_loss": max_loss,
        "final_loss": final_loss,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "spike_steps": spike_steps,
        "spike_count": spike_count,
        "plateau": plateau,
        "divergence": divergence,
        "flags": flags,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect NaN/Inf, spike, plateau, and divergence flags on a "
            "training loss curve."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="CSV with header 'step,loss'")
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    curve = load_csv(args.input)
    result = detect(curve)

    print("Loss curve analysis")
    print("===================")
    print(f"  steps        : {result['steps']}")
    print(f"  min loss     : {result['min_loss']:.4f}")
    print(f"  max loss     : {result['max_loss']:.4f}")
    print(f"  final loss   : {result['final_loss']:.4f}")
    print(f"  nan count    : {result['nan_count']}")
    print(f"  inf count    : {result['inf_count']}")
    print(f"  spike count  : {result['spike_count']}")
    if result["spike_count"] > 0:
        print(f"  top spikes   : {result['spike_steps']}")
    print("FLAGS: " + " ".join(result["flags"]))

    print()
    print("Flag details")
    print("------------")
    print(f"  NAN        : {result['nan_count']} rows")
    print(f"  INF        : {result['inf_count']} rows")
    print(f"  SPIKE      : {result['spike_count']} steps "
          f"(|loss - median(11)| > 5*MAD(11))")
    if result["spike_count"] > 0:
        print(f"              top spike steps: {result['spike_steps']}")
    tail = curve[-PLATEAU_WINDOW:]
    finite_tail = [(step, loss) for step, loss in tail if math.isfinite(loss)]
    slope = (linear_regression(
        [(float(step), loss) for step, loss in finite_tail]
    ) if len(finite_tail) >= 2 else float("nan"))
    threshold = (
        f"final {result['final_loss']:.4f} > "
        f"1.5*min {result['min_loss']:.4f}"
    )
    print(f"  PLATEAU    : slope={slope:.5f}/step (last "
          f"{len(finite_tail)} rows), {threshold} -> {result['plateau']}")
    recent = [(step, loss) for step, loss in curve[-DIVERGENCE_WINDOW:]
              if math.isfinite(loss)]
    monotonic = (
        all(recent[i][1] >= recent[i - 1][1] for i in range(1, len(recent)))
        if len(recent) >= 2 else False
    )
    print(f"  DIVERGENCE : last {len(recent)} finite rows monotonic "
          f"non-decreasing -> {monotonic}")


if __name__ == "__main__":
    main()
