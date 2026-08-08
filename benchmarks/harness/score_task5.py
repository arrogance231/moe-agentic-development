#!/usr/bin/env python3
"""Deterministic scorer for Task 5 (Conflicting-guidance resolution).

Checks internal consistency against the pre-registered incompatible-pair
list in task_prompts.INCOMPATIBLE_PAIRS, plus a 0-5 justification-quality
proxy (presence of an explicit numeric choice + a justification clause per
parameter). Per methodology.md, justification quality proper requires human/
independent-reviewer judgment; this automated proxy is a floor, not a
replacement, and is documented as such.
"""
import argparse
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task_prompts import INCOMPATIBLE_PAIRS  # noqa: E402


def score(text: str) -> dict:
    violations = []
    for pair in INCOMPATIBLE_PAIRS:
        if "routing_pattern" in pair:
            if re.search(pair["routing_pattern"], text, re.I) and re.search(
                pair["conflicting_pattern"], text, re.I
            ):
                violations.append(pair["id"])
        elif pair["id"] == "capacity_factor_out_of_stated_range":
            m = re.search(r"capacity[_ ]factor[^0-9]{0,20}(\d+\.?\d*)", text, re.I)
            if m and not (1.0 <= float(m.group(1)) <= 2.0) and "justif" not in text.lower():
                violations.append(pair["id"])
        elif pair["id"] == "aux_loss_coefficient_out_of_stated_range":
            m = re.search(r"aux(?:iliary)?[- ]loss[^0-9]{0,30}(\d+\.?\d*)", text, re.I)
            if m and not (0.001 <= float(m.group(1)) <= 0.1) and "justif" not in text.lower():
                violations.append(pair["id"])

    internal_consistency = 1 if not violations else 0

    params = ["capacity factor", "aux", "top-k", "top_k"]
    stated = sum(1 for p in params if re.search(re.escape(p), text, re.I))
    justif_words = len(re.findall(r"because|since|due to|justif", text, re.I))
    revise_condition = bool(re.search(r"revisit|revise|reconsider|deviat", text, re.I))
    justification_quality = min(5, stated + (1 if justif_words >= 2 else 0) + (1 if revise_condition else 0))

    return {
        "internal_consistency": internal_consistency,
        "violations": violations,
        "justification_quality": justification_quality,
        "revise_condition_present": revise_condition,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    with open(args.proposal) as f:
        text = f.read()
    result = score(text)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result)
