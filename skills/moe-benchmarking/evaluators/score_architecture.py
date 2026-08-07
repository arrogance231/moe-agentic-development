"""score_architecture.py -- score a proposed MoE architecture document against a rubric.

Pure Python 3 stdlib only (argparse, json, re, sys, pathlib). Deterministic:
the same proposal + rubric always yields the same scores.

Built-in rubric
---------------
Five criteria, each scored 0-5 via deterministic text heuristics. Every
criterion is a boolean "success condition" plus a list of partial-credit
sub-elements. Generic scoring rule:

    0  if the required elements are entirely absent.
    5  if ALL required elements are present (the success condition holds).
    1-4 one point per satisfied sub-element otherwise.

The five criteria:

1. param_math -- correctness of parameter math. Success condition: all five
   sub-elements present. Sub-elements:
     a. table_block      a table-like block containing digit counts (a line
                         containing "|" and at least one digit).
     b. total_params     an explicit total-parameter figure ("1B", "3.2B",
                         "1,000,000", "8_000_000").
     c. num_experts      an expert-count number ("num_experts: 32", "32 experts").
     d. top_k            a top-k number ("top_k: 2", "top k = 1", "Top-2").
     e. consistency      the same expert count is mentioned at least twice.

2. efficiency -- the proposed config is efficient. Success condition: expert
   count in 16-64 AND top-k in 1-2. Sub-elements:
     a. experts_present   an expert count was found.
     b. experts_in_range  expert count in 16-64.
     c. topk_present      a top-k value was found.
     d. topk_in_range     top-k in 1-2.

3. expert_utilization_awareness -- mentions capacity factor and/or load
   balancing. Success condition: ("capacity factor" or "capacity_factor") AND
   ("aux loss" or "load-balanc*"). Sub-elements:
     a. capacity_factor   "capacity factor" / "capacity_factor".
     b. aux_loss          "aux loss".
     c. load_balance      "load-balanc" / "load_balanc" / "load balanc".

4. completeness -- has the expected-output sections. Success condition: >= 4 of
   the 5 section keywords present. Sub-elements (keywords):
   overview, parameters, routing, training implications, risks.

5. justification -- routing choice justified. Success condition: a routing
   keyword AND a justification word. Sub-elements:
     a. routing_keyword    "Top-1"/"Top-2"/"top-1"/"top-2"/"learned"/"soft".
     b. justification_word "because"/"since"/"due to"/"justif".

Custom rubric (--rubric path.json)
----------------------------------
Optional JSON file with this schema:

    {
      "criteria": {
        "param_math": ["regex1", "regex2", ...],
        "efficiency": ["..."],
        ...
      },
      "pass_threshold": 15,
      "total_points": 25
    }

Each criterion maps a name to a list of regex patterns. A criterion scores 5
when every pattern matches the text (re.search, case-insensitive); otherwise it
scores one point per matching pattern (capped at 4). Criteria are evaluated in
the order they appear in the file. "pass_threshold" defaults to 15 and
"total_points" defaults to 25.

Output
------
Per-criterion scores, total (out of 25 by default), and PASS/FAIL
(PASS if total >= 15 by default). Plain text by default; --json emits the full
result object.

Importable API: score_text(text, rubric: dict | None = None) -> dict,
load_rubric(path) -> dict.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_PASS_THRESHOLD = 15
DEFAULT_TOTAL_POINTS = 25

_EXPERT_PATTERNS = [
    re.compile(r"num[_\s]?experts\s*[=:]\s*(\d+)", re.IGNORECASE),
    re.compile(r"experts\s*[=:]\s*(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)\s+(?:total\s+)?experts\b", re.IGNORECASE),
]

_TOPK_PATTERNS = [
    re.compile(r"top[_\s-]?k\s*[=:]\s*(\d+)", re.IGNORECASE),
    re.compile(r"top[_\s-]?k\s+(\d+)", re.IGNORECASE),
    re.compile(r"\btop\s*-\s*(\d)\b", re.IGNORECASE),
]

_TOTAL_PARAMS_PATTERNS = [
    re.compile(r"\b\d+(?:\.\d+)?\s*[Bb]\b"),
    re.compile(r"\b\d{1,3}(?:,\d{3})+\b"),
    re.compile(r"\b\d+_\d{3,}\b"),
]


def _extract_expert_counts(text: str) -> list[int]:
    counts = []
    for pattern in _EXPERT_PATTERNS:
        counts.extend(int(m) for m in pattern.findall(text))
    return counts


def _extract_topk(text: str) -> list[int]:
    values = []
    for pattern in _TOPK_PATTERNS:
        values.extend(int(m) for m in pattern.findall(text))
    return values


def _count_standalone(value: int, text: str) -> int:
    pattern = re.compile(r"(?<![0-9.])" + str(value) + r"(?![0-9.])")
    return len(pattern.findall(text))


def _has_table_block(text: str) -> bool:
    for line in text.splitlines():
        if "|" in line and any(ch.isdigit() for ch in line):
            return True
    return False


def _has_total_params(text: str) -> bool:
    return any(p.search(text) is not None for p in _TOTAL_PARAMS_PATTERNS)


def _param_math(text: str) -> tuple[bool, dict]:
    experts = _extract_expert_counts(text)
    topk = _extract_topk(text)
    consistent = any(_count_standalone(v, text) >= 2 for v in set(experts))
    elements = {
        "table_block": _has_table_block(text),
        "total_params": _has_total_params(text),
        "num_experts": bool(experts),
        "top_k": bool(topk),
        "consistency": consistent,
    }
    return all(elements.values()), elements


def _efficiency(text: str) -> tuple[bool, dict]:
    experts = _extract_expert_counts(text)
    topk = _extract_topk(text)
    experts_in_range = any(16 <= v <= 64 for v in experts)
    topk_in_range = any(1 <= v <= 2 for v in topk)
    elements = {
        "experts_present": bool(experts),
        "experts_in_range": experts_in_range,
        "topk_present": bool(topk),
        "topk_in_range": topk_in_range,
    }
    return experts_in_range and topk_in_range, elements


def _expert_utilization_awareness(text: str) -> tuple[bool, dict]:
    lower = text.lower()
    capacity = "capacity factor" in lower or "capacity_factor" in lower
    aux_loss = "aux loss" in lower
    load_balance = (
        "load-balanc" in lower or "load_balanc" in lower or "load balanc" in lower
    )
    elements = {
        "capacity_factor": capacity,
        "aux_loss": aux_loss,
        "load_balance": load_balance,
    }
    return capacity and (aux_loss or load_balance), elements


def _completeness(text: str) -> tuple[bool, dict]:
    lower = text.lower()
    keywords = {
        "overview": "overview" in lower,
        "parameters": "parameters" in lower,
        "routing": "routing" in lower,
        "training implications": "training implication" in lower,
        "risks": "risks" in lower,
    }
    present = sum(1 for v in keywords.values() if v)
    return present >= 4, keywords


def _justification(text: str) -> tuple[bool, dict]:
    lower = text.lower()
    routing_keyword = any(kw in text for kw in ("Top-1", "Top-2", "top-1", "top-2"))
    routing_keyword = routing_keyword or (
        re.search(r"\blearned\b", text) is not None
        or re.search(r"\bsoft\b", text) is not None
    )
    justification_word = any(w in lower for w in ("because", "since", "due to", "justif"))
    elements = {
        "routing_keyword": routing_keyword,
        "justification_word": justification_word,
    }
    return routing_keyword and justification_word, elements


_BUILTIN_CRITERIA = (
    ("param_math", _param_math),
    ("efficiency", _efficiency),
    ("expert_utilization_awareness", _expert_utilization_awareness),
    ("completeness", _completeness),
    ("justification", _justification),
)


def _score_rule(condition: bool, elements: dict) -> int:
    if condition:
        return 5
    return min(sum(1 for v in elements.values() if v), 4)


def score_text(text: str, rubric: dict | None = None) -> dict:
    """Score architecture proposal text against a rubric.

    Returns a dict with keys "criteria" (per-criterion score dict), "total",
    "passed", plus "total_points" and "threshold" for reporting.
    """
    if rubric is None:
        criteria = {}
        for name, fn in _BUILTIN_CRITERIA:
            condition, elements = fn(text)
            criteria[name] = {"score": _score_rule(condition, elements), "elements": elements}
        total_points = DEFAULT_TOTAL_POINTS
        threshold = DEFAULT_PASS_THRESHOLD
    else:
        criteria = {}
        raw = rubric.get("criteria") or {}
        for name, patterns in raw.items():
            if not isinstance(patterns, list):
                raise ValueError(f"criterion {name!r} must map to a list of regex strings")
            matches = {p: re.search(p, text, re.IGNORECASE) is not None for p in patterns}
            condition = all(matches.values())
            criteria[name] = {"score": _score_rule(condition, matches), "elements": matches}
        total_points = int(rubric.get("total_points", DEFAULT_TOTAL_POINTS))
        threshold = int(rubric.get("pass_threshold", DEFAULT_PASS_THRESHOLD))

    total = sum(c["score"] for c in criteria.values())
    return {
        "criteria": criteria,
        "total": total,
        "passed": total >= threshold,
        "total_points": total_points,
        "threshold": threshold,
    }


def load_rubric(path: str | Path) -> dict:
    """Load and validate a custom rubric JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        rubric = json.load(fh)
    if not isinstance(rubric, dict):
        raise ValueError("rubric file must contain a JSON object")
    if "criteria" not in rubric:
        raise ValueError(
            "rubric file must include a 'criteria' object mapping criterion names "
            "to lists of regex patterns"
        )
    if not isinstance(rubric["criteria"], dict):
        raise ValueError("rubric['criteria'] must be an object")
    return rubric


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score an MoE architecture proposal document against a rubric."
    )
    parser.add_argument(
        "--proposal", required=True, metavar="path.md",
        help="Path to the architecture proposal (markdown/text).",
    )
    parser.add_argument(
        "--rubric", metavar="path.json",
        help="Optional custom rubric JSON file (default: built-in rubric).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the full result as JSON.",
    )
    args = parser.parse_args(argv)

    proposal_path = Path(args.proposal)
    if not proposal_path.is_file():
        parser.error(f"proposal file not found: {proposal_path}")
    text = proposal_path.read_text(encoding="utf-8", errors="replace")

    try:
        rubric = load_rubric(args.rubric) if args.rubric else None
        result = score_text(text, rubric)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"proposal: {proposal_path}")
        for name, entry in result["criteria"].items():
            print(f"  {name}: {entry['score']}")
        print(f"total: {result['total']} / {result['total_points']}")
        print(f"threshold: {result['threshold']}")
        print(f"result: {'PASS' if result['passed'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
