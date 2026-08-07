#!/usr/bin/env python3
"""Aggregate the moe-debugging analyzers into one markdown report.

Purpose
-------
Runs the router-distribution, loss-analysis, and expert-utilization analyzers
on their respective CSVs and composes a single markdown diagnosis report with
a synthesis section that maps the strongest observed flag to a likely MoE
failure class.

Synthesis priority
------------------
When multiple flags fire, the strongest signal is chosen in this order, so
that the most severe / root-cause-level failure class wins:

    1. Numerical   -- NAN, INF, SPIKE, or DIVERGENCE (loss analyzers)
    2. Collapse    -- COLLAPSED (router distribution)
    3. Overflow    -- OVERFLOW (router distribution or expert utilization)
    4. Imbalance   -- IMBALANCED or UNDERUTILIZED
    5. (none)      -- no strong signal

Usage
-----
    python3 diagnosis_report.py --router counts.csv --loss loss.csv \
        --expert counts.csv [--capacity-factor 1.0] [--output report.md]

The report is written to ``--output`` when given, otherwise printed to
stdout. Exit code 0 on success.
"""

import argparse
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

from analyzers import (  # noqa: E402  (sys.path setup above)
    expert_utilization,
    loss_analyzer,
    router_distribution,
)


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


def failure_class(flags: set[str]) -> tuple[str, str | None]:
    """Map the strongest flag to (failure class, SKILL.md workflow anchor).

    Priority: numerical > collapse > overflow > imbalance (see module
    docstring). Returns ('none', None) when no flag fires.
    """
    numerical = {"NAN", "INF", "SPIKE", "DIVERGENCE"} & flags
    if numerical:
        return (
            "Numerical instability",
            "`### Exploding loss / NaN`",
        )
    if "COLLAPSED" in flags:
        return ("Router collapse", "`### Router collapse`")
    if "OVERFLOW" in flags:
        return (
            "Capacity overflow / token dropping",
            "`### Router collapse`",
        )
    if {"IMBALANCED", "UNDERUTILIZED"} & flags:
        return ("Expert imbalance", "`### Expert imbalance`")
    return ("none", None)


def render_router_section(result: dict) -> str:
    """Render section 1 (router distribution) as markdown."""
    lines = [
        "## 1. Router distribution",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total tokens | {result['total']:,} |",
        f"| Experts (n) | {result['n']} |",
        f"| Normalized entropy | {result['entropy_norm']:.4f} |",
        f"| Gini coefficient | {result['gini']:.4f} |",
        f"| Top-expert share | {result['top_share']:.4f} |",
        f"| Effective experts | {result['effective_experts']:.2f} |",
        f"| Utilization skew | {result['skew']:.2f} |",
        f"| Overflow fraction | {result['overflow_fraction']:.4f} |",
        "",
    ]
    return "\n".join(lines)


def render_loss_section(result: dict) -> str:
    """Render section 2 (loss analysis) as markdown."""
    lines = [
        "## 2. Loss analysis",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Steps | {result['steps']} |",
        f"| Min loss | {result['min_loss']:.4f} |",
        f"| Max loss | {result['max_loss']:.4f} |",
        f"| Final loss | {result['final_loss']:.4f} |",
        f"| NaN rows | {result['nan_count']} |",
        f"| Inf rows | {result['inf_count']} |",
        f"| Spike count | {result['spike_count']} |",
        f"| Top spike steps | {result['spike_steps']} |",
        f"| Plateau | {result['plateau']} |",
        f"| Divergence | {result['divergence']} |",
        "",
    ]
    return "\n".join(lines)


def render_expert_section(result: dict) -> str:
    """Render section 3 (expert utilization) as markdown."""
    lines = [
        "## 3. Expert utilization",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total tokens | {result['total']:,} |",
        f"| Experts (n) | {result['n']} |",
        f"| Utilization % | {result['utilization_pct']:.1f}% |",
        f"| Skew (max/min) | {result['skew']:.2f} |",
        f"| Overflow experts | {result['overflow_experts']} |",
        f"| Overflow token fraction | "
        f"{result['overflow_token_fraction']:.4f} |",
        "",
        "| Expert | Count | Share % | Capacity |",
        "| --- | --- | --- | --- |",
    ]
    for row in result["per_expert"]:
        lines.append(
            f"| e{row['expert']:02d} | {row['count']:,} | "
            f"{row['share']:.1f}% | {row['marker']} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_report(
    router: dict, loss: dict, expert: dict, capacity_factor: float
) -> str:
    """Compose the full markdown diagnosis report."""
    all_flags = set(router["flags"]) | set(loss["flags"]) | set(expert["flags"])
    class_name, anchor = failure_class(all_flags)

    sections = [
        "# MoE Training Diagnosis Report",
        "",
        "## 1. Router distribution",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total tokens | {router['total']:,} |",
        f"| Experts (n) | {router['n']} |",
        f"| Normalized entropy | {router['entropy_norm']:.4f} |",
        f"| Gini coefficient | {router['gini']:.4f} |",
        f"| Top-expert share | {router['top_share']:.4f} |",
        f"| Effective experts | {router['effective_experts']:.2f} |",
        f"| Utilization skew | {router['skew']:.2f} |",
        f"| Overflow fraction | {router['overflow_fraction']:.4f} |",
        "",
        "## 2. Loss analysis",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Steps | {loss['steps']} |",
        f"| Min loss | {loss['min_loss']:.4f} |",
        f"| Max loss | {loss['max_loss']:.4f} |",
        f"| Final loss | {loss['final_loss']:.4f} |",
        f"| NaN rows | {loss['nan_count']} |",
        f"| Inf rows | {loss['inf_count']} |",
        f"| Spike count | {loss['spike_count']} |",
        f"| Top spike steps | {loss['spike_steps']} |",
        f"| Plateau | {loss['plateau']} |",
        f"| Divergence | {loss['divergence']} |",
        "",
        "## 3. Expert utilization",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total tokens | {expert['total']:,} |",
        f"| Experts (n) | {expert['n']} |",
        f"| Utilization % | {expert['utilization_pct']:.1f}% |",
        f"| Skew (max/min) | {expert['skew']:.2f} |",
        f"| Overflow experts | {expert['overflow_experts']} |",
        f"| Overflow token fraction | "
        f"{expert['overflow_token_fraction']:.4f} |",
        "",
        "| Expert | Count | Share % | Capacity |",
        "| --- | --- | --- | --- |",
    ]
    for row in expert["per_expert"]:
        sections.append(
            f"| e{row['expert']:02d} | {row['count']:,} | "
            f"{row['share']:.1f}% | {row['marker']} |"
        )

    sections.append("")
    sections.append("## 4. Synthesis")
    sections.append("")
    sections.append("All flags observed across analyzers:")
    sections.append("")
    for flag in sorted(all_flags):
        sections.append(f"- `{flag}`")
    sections.append("")
    sections.append(
        f"**Strongest signal:** {class_name}."
    )
    if anchor:
        sections.append(
            f"**Recommendation:** apply the {anchor} workflow in SKILL.md."
        )
    else:
        sections.append(
            "**Recommendation:** none of the monitored failure signatures "
            "fired; continue normal training and re-run on the next "
            "checkpoint."
        )
    sections.append("")
    sections.append(
        f"_capacity-factor used: {capacity_factor:g}; diagnosis is "
        "evidence-driven, not a training change._"
    )
    sections.append("")
    return "\n".join(sections)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate router, loss, and expert-utilization analyzers into "
            "a markdown MoE diagnosis report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--router", required=True,
                        help="router CSV with header 'expert,count' or "
                             "'expert,probability'")
    parser.add_argument("--loss", required=True,
                        help="loss CSV with header 'step,loss'")
    parser.add_argument("--expert", required=True,
                        help="expert CSV with header 'expert,count'")
    parser.add_argument("--capacity-factor", type=positive_float, default=1.0,
                        help="expert capacity as a multiple of the balanced "
                             "load (total/n)")
    parser.add_argument("--output", default=None,
                        help="write the report to this file instead of "
                             "stdout")
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()

    _, router_counts = router_distribution.load_csv(args.router)
    router = router_distribution.analyze(
        router_counts, args.capacity_factor
    )

    loss_curve = loss_analyzer.load_csv(args.loss)
    loss = loss_analyzer.detect(loss_curve)

    _, expert_counts = expert_utilization.load_csv(args.expert)
    expert = expert_utilization.analyze(
        expert_counts, args.capacity_factor
    )

    report = build_report(router, loss, expert, args.capacity_factor)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
