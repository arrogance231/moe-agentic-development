#!/usr/bin/env python3
"""Score all 120 benchmark outputs and emit a flat JSON dataset."""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.expanduser("~/moe-agentic-development")
EVAL = os.path.join(ROOT, "skills/moe-benchmarking/evaluators")
HARNESS = os.path.join(ROOT, "benchmarks/harness")
RESULTS = os.path.join(ROOT, "benchmarks/results/run-20260808-0559")
GT3 = os.path.join(ROOT, "benchmarks/results/gt_task3.json")

ARM_DIR = {
    "A0": "benchmarks/baseline/A0",
    "A1": "benchmarks/baseline/A1",
    "A2": "benchmarks/with-skills/A2",
    "A3": "benchmarks/with-skills/A3",
}
EXT = {"task1": "md", "task2": "json", "task3": "json", "task4": "md", "task5": "md", "task6": "md"}
ARMS = ["A0", "A1", "A2", "A3"]
TASKS = [f"task{i}" for i in range(1, 7)]
SEEDS = [1, 2, 3, 4, 5]

failures_log = []
rows = []


def extract_json_block(text):
    """Extract the first valid JSON object from prose/fenced markdown text."""
    text = text.strip()
    try:
        return json.loads(text), text
    except Exception:
        pass
    # fenced ```json ... ``` block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1)), m.group(1)
        except Exception:
            pass
    # first balanced {...} anywhere (greedy brace matching)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate), candidate
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None, None


def normalize_json_file(path):
    """Return (parsed_json_or_None, raw_json_text_or_None) extracted from a
    task2/task3 output file that may contain prose + fenced JSON."""
    with open(path) as f:
        raw = f.read()
    return extract_json_block(raw)


def run_score_architecture(path):
    out = subprocess.run(
        [sys.executable, os.path.join(EVAL, "score_architecture.py"), "--proposal", path, "--json"],
        capture_output=True, text=True,
    )
    return json.loads(out.stdout) if out.returncode == 0 else None


def run_score_debugging(path):
    parsed, raw_json_text = normalize_json_file(path)
    if parsed is None:
        return None
    tmp = "/tmp/_t3_score_tmp.json"
    with open(tmp, "w") as f:
        f.write(raw_json_text)
    out = subprocess.run(
        [sys.executable, os.path.join(EVAL, "score_debugging.py"), "--diagnosis", tmp,
         "--ground-truth", GT3, "--json"],
        capture_output=True, text=True,
    )
    return json.loads(out.stdout) if out.returncode == 0 else None


def run_score_task5(path):
    out = subprocess.run(
        [sys.executable, os.path.join(HARNESS, "score_task5.py"), "--proposal", path, "--json"],
        capture_output=True, text=True,
    )
    return json.loads(out.stdout) if out.returncode == 0 else None


NUMKEY_RE = {
    "estimated_tokens_per_sec": re.compile(r"estimated_tokens_per_sec"),
}


def score_task2(path):
    try:
        d, _ = normalize_json_file(path)
        if d is None:
            return {"successful_launch": 0, "error": "no parseable JSON object found"}
    except Exception as e:
        return {"successful_launch": 0, "error": str(e)}
    required = ["framework", "architecture", "parallelism", "batch_size",
                "estimated_memory_gb", "estimated_tokens_per_sec", "launch_notes"]
    ok = all(k in d for k in required)
    arch_ok = False
    mem_ratio = None
    tok_s = None
    if ok:
        arch = d.get("architecture", {})
        par = d.get("parallelism", {})
        arch_keys = ["num_experts", "top_k", "hidden_size", "num_layers", "capacity_factor"]
        par_keys = ["data_parallel", "expert_parallel", "tensor_parallel"]
        arch_ok = all(k in arch for k in arch_keys) and all(k in par for k in par_keys)
        if arch_ok:
            try:
                arch_ok = arch["top_k"] <= arch["num_experts"]
            except Exception:
                arch_ok = False
        try:
            ngpu = par["data_parallel"] * par["expert_parallel"] * par["tensor_parallel"]
            mem_ratio = d["estimated_memory_gb"] / (ngpu * 80) if ngpu else None
        except Exception:
            mem_ratio = None
        tok_s = d.get("estimated_tokens_per_sec")
    successful = 1 if (ok and arch_ok) else 0
    return {"successful_launch": successful, "memory_ratio": mem_ratio, "tokens_per_sec": tok_s}


def _numbers_in_row(row_text):
    """Extract numeric tokens (with optional ~, commas) from a markdown table row."""
    nums = []
    for m in re.finditer(r"~?\s*(\d[\d,]*\.?\d*)", row_text):
        try:
            nums.append(float(m.group(1).replace(",", "")))
        except Exception:
            pass
    return nums


def score_task4(text):
    after_toks = None
    after_util = None
    for line in text.splitlines():
        if re.search(r"token", line, re.I) and "|" in line and re.search(r"\d", line):
            nums = _numbers_in_row(line)
            # first number close to 12400 is "before"; take a later, different number as "after"
            candidates = [n for n in nums if n > 100]  # filter out stray small numbers/deltas
            if len(candidates) >= 2:
                after_toks = candidates[1]
            elif len(candidates) == 1 and abs(candidates[0] - 12400) > 1:
                after_toks = candidates[0]
        if re.search(r"utili[sz]ation", line, re.I) and "|" in line and "%" in line:
            pcts = [float(v) for v in re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", line)]
            pcts = [v for v in pcts if 0 <= v <= 100]
            if len(pcts) >= 2:
                after_util = pcts[1]
            elif len(pcts) == 1 and abs(pcts[0] - 62) > 0.5:
                after_util = pcts[0]
    delta = ((after_toks - 12400) / 12400) if after_toks else None
    util_delta = (after_util - 62) if after_util is not None else None
    return {"claimed_after_tok_s": after_toks, "throughput_delta_frac": delta,
            "claimed_after_util_pct": after_util, "util_delta_pct": util_delta}


def score_task6(text):
    arch = run_score_architecture_text(text)
    ep4 = bool(re.search(r"expert.?parallel.{0,20}4|EP\s*[:=]?\s*4|EP degree\s*[:=]?\s*4", text, re.I))
    commvol = bool(re.search(r"communication|comm\.?\s*volume|bandwidth|traffic", text, re.I))
    gbps = bool(re.search(r"25\s*Gbps|Gbps", text, re.I))
    constraint_fit = 1 if (ep4 and commvol and gbps) else 0
    return {"architecture": arch, "constraint_fit": constraint_fit}


def run_score_architecture_text(text):
    tmp = "/tmp/_t6_score_tmp.md"
    with open(tmp, "w") as f:
        f.write(text)
    return run_score_architecture(tmp)


for arm in ARMS:
    d = os.path.join(ROOT, ARM_DIR[arm])
    for task in TASKS:
        ext = EXT[task]
        for seed in SEEDS:
            base = f"{task}_run{seed}"
            path = os.path.join(d, f"{base}.{ext}")
            metrics_path = os.path.join(d, f"{base}.metrics.json")
            row = {"arm": arm, "task": task, "seed": seed}
            if not os.path.exists(path):
                failures_log.append(f"MISSING OUTPUT: {arm}/{base}.{ext}")
                rows.append(row)
                continue
            try:
                with open(metrics_path) as f:
                    m = json.load(f)
                row["metrics"] = m
            except Exception as e:
                failures_log.append(f"MISSING/BAD METRICS: {arm}/{base}.metrics.json: {e}")
                row["metrics"] = {}

            try:
                if task == "task1":
                    row["score"] = run_score_architecture(path)
                elif task == "task2":
                    row["score"] = score_task2(path)
                elif task == "task3":
                    row["score"] = run_score_debugging(path)
                elif task == "task4":
                    with open(path) as f:
                        text = f.read()
                    row["score"] = score_task4(text)
                elif task == "task5":
                    row["score"] = run_score_task5(path)
                elif task == "task6":
                    with open(path) as f:
                        text = f.read()
                    row["score"] = score_task6(text)
            except Exception as e:
                failures_log.append(f"SCORING FAILURE: {arm}/{base}: {type(e).__name__}: {e}")
                row["score"] = None
            rows.append(row)

os.makedirs(RESULTS, exist_ok=True)
with open(os.path.join(RESULTS, "scored_dataset.json"), "w") as f:
    json.dump(rows, f, indent=2)

if failures_log:
    with open(os.path.join(RESULTS, "failures.md"), "a") as f:
        import datetime
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        f.write(f"\n{ts} — SCORING PHASE: {len(failures_log)} scoring issue(s) found:\n")
        for line in failures_log:
            f.write(f"{ts} — {line}\n")

print(f"Scored {len(rows)} rows, {len(failures_log)} scoring issues")
print("Sample row:", json.dumps(rows[0], indent=2)[:500])
