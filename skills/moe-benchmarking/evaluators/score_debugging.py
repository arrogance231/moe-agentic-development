"""score_debugging.py -- compare an agent's debugging diagnosis against ground truth.

Pure Python 3 stdlib only (argparse, json, re, sys). Deterministic: the same
diagnosis and ground truth always yield the same scores.

Ground-truth JSON schema (--ground-truth):
    {
      "problem": "router collapse",
      "root_cause": "aux loss too low",
      "fix": "increase aux loss to 0.01",
      "key_evidence": ["top expert share > 0.5", "entropy < 0.3"]
    }

Diagnosis JSON schema (--diagnosis):
    {
      "identified_problems": ["expert collapse"],
      "root_causes": ["auxiliary loss weight too low"],
      "actions": ["raise aux loss coefficient to 0.01"],
      "evidence": ["top expert share > 0.5", "entropy < 0.3"]
    }

Metrics (all deterministic):

- problem_detected: 1.0 if any identified problem shares a keyword (lowercase
  alphanumeric token) with the ground-truth problem; else 0.0.
- root_cause_match: Jaccard similarity (intersection over union) between the
  diagnosis root_causes token set and the ground-truth root_cause token set.
- fix_match: Jaccard similarity between the diagnosis actions token set and the
  ground-truth fix token set.
- evidence_cited: fraction of ground-truth key_evidence items matched by any
  diagnosis evidence string (case-insensitive substring match).
- accuracy: weighted sum
        0.30 * problem_detected
      + 0.30 * root_cause_match
      + 0.25 * fix_match
      + 0.15 * evidence_cited

Tokens are lowercase runs of alphanumeric characters. Jaccard on two empty sets
is defined as 0.0. evidence_cited is 1.0 when ground truth lists no
key_evidence items (vacuously all cited). All metrics are reported with 3
decimals. The CLI exits 0 on success and 1 on file/JSON errors.

Importable API: load_json(path), tokenize(s), jaccard(a, b),
score(diagnosis: dict, ground_truth: dict) -> dict.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_json(path: str) -> dict:
    """Read a JSON file and return its parsed object."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def tokenize(s) -> list[str]:
    """Lowercase alphanumeric tokens of a string (non-token chars are separators)."""
    return _TOKEN_RE.findall(str(s).lower())


def jaccard(a, b) -> float:
    """Jaccard similarity of two token sequences; 0.0 when both are empty."""
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def score(diagnosis: dict, ground_truth: dict) -> dict:
    """Score a diagnosis against ground truth.

    Returns a dict with the five metrics, each rounded to 3 decimals.
    """
    gt_problem = str(ground_truth.get("problem", ""))
    gt_root_cause = str(ground_truth.get("root_cause", ""))
    gt_fix = str(ground_truth.get("fix", ""))
    gt_evidence = [str(e) for e in (ground_truth.get("key_evidence") or [])]

    problems = [str(p) for p in (diagnosis.get("identified_problems") or [])]
    root_causes = [str(r) for r in (diagnosis.get("root_causes") or [])]
    actions = [str(a) for a in (diagnosis.get("actions") or [])]
    evidence = [str(e) for e in (diagnosis.get("evidence") or [])]

    gt_problem_tokens = set(tokenize(gt_problem))
    problem_detected = 1.0 if any(
        set(tokenize(p)) & gt_problem_tokens for p in problems
    ) else 0.0

    rc_tokens = set()
    for rc in root_causes:
        rc_tokens |= set(tokenize(rc))
    root_cause_match = jaccard(rc_tokens, set(tokenize(gt_root_cause)))

    action_tokens = set()
    for act in actions:
        action_tokens |= set(tokenize(act))
    fix_match = jaccard(action_tokens, set(tokenize(gt_fix)))

    gt_evidence_lower = [e.lower() for e in gt_evidence]
    evidence_lower = [e.lower() for e in evidence]
    matched = sum(
        1 for item in gt_evidence_lower if any(item in d for d in evidence_lower)
    )
    evidence_cited = matched / len(gt_evidence_lower) if gt_evidence_lower else 1.0

    accuracy = (
        0.30 * problem_detected
        + 0.30 * root_cause_match
        + 0.25 * fix_match
        + 0.15 * evidence_cited
    )

    return {
        "problem_detected": round(problem_detected, 3),
        "root_cause_match": round(root_cause_match, 3),
        "fix_match": round(fix_match, 3),
        "evidence_cited": round(evidence_cited, 3),
        "accuracy": round(accuracy, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score an agent's debugging diagnosis against ground truth."
    )
    parser.add_argument(
        "--diagnosis", required=True, metavar="path.json",
        help="Agent diagnosis JSON file.",
    )
    parser.add_argument(
        "--ground-truth", required=True, metavar="path.json",
        help="Ground-truth JSON file.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the result as JSON.",
    )
    args = parser.parse_args(argv)

    try:
        diagnosis = load_json(args.diagnosis)
        ground_truth = load_json(args.ground_truth)
        result = score(diagnosis, ground_truth)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for name, value in result.items():
            print(f"{name}: {value:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
