#!/usr/bin/env python3
"""MoE benchmark run harness.

Invokes OpenCode CLI (model opencode/deepseek-v4-flash-free) for one
(arm, task, seed) triple, per skills/moe-benchmarking/examples/benchmark-protocol.md.

Usage:
    python run_harness.py --arm A1 --task task1 --seed 1 [--run-id run-XXXX]

Arm -> retrieval/skills mapping and confound controls (search disabled at the
*harness* level via per-arm opencode.jsonc, never by prompt instruction) are
documented in benchmarks/harness/configs/<ARM>/opencode.jsonc.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from task_prompts import TASKS  # noqa: E402

OPENCODE_BIN = os.path.expanduser("~/.opencode/bin/opencode")
MODEL = "opencode/deepseek-v4-flash-free"

ARM_DIRS = {
    "A0": ("benchmarks/baseline/A0", False),
    "A1": ("benchmarks/baseline/A1", True),
    "A2": ("benchmarks/with-skills/A2", False),
    "A3": ("benchmarks/with-skills/A3", True),
}
SKILLS_ARMS = {"A2", "A3"}


def load_skill_text():
    skill_md = os.path.join(REPO_ROOT, "skills", "moe-benchmarking", "..", "moe-training", "SKILL.md")
    # Use whichever MoE-domain skill(s) exist under skills/, excluding the
    # meta 'moe-benchmarking' skill itself (that's the evaluator, not domain
    # knowledge the agent should be handed).
    skills_dir = os.path.join(REPO_ROOT, "skills")
    texts = []
    for name in sorted(os.listdir(skills_dir)):
        if name == "moe-benchmarking":
            continue
        path = os.path.join(skills_dir, name, "SKILL.md")
        if os.path.isfile(path):
            with open(path) as f:
                texts.append(f"<!-- skill: {name} -->\n" + f.read())
    return "\n\n".join(texts)


def build_prompt(task_id, arm):
    task = TASKS[task_id]
    prompt = task["prompt"]
    if arm in SKILLS_ARMS:
        skill_text = load_skill_text()
        if skill_text:
            prompt = (
                "You have the following procedural domain skills available as "
                "reference material. They do not instruct you to search; use "
                "them as fixed procedural guidance.\n\n"
                f"{skill_text}\n\n---\n\nTask:\n{prompt}"
            )
    return prompt


def run(arm, task_id, seed, run_id, out_root):
    task = TASKS[task_id]
    out_dir, search_enabled = ARM_DIRS[arm]
    out_dir_abs = os.path.join(REPO_ROOT, out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)

    config_dir = os.path.join(HERE, "configs", arm)
    prompt = build_prompt(task_id, arm)

    query_log = os.path.join(out_dir_abs, f"{task_id}_run{seed}.queries.jsonl")
    if os.path.exists(query_log):
        os.remove(query_log)

    env = dict(os.environ)
    env["MOE_BENCH_QUERY_LOG"] = query_log
    env["OPENCODE_DISABLE_AUTOUPDATE"] = "1"

    cmd = [
        OPENCODE_BIN, "run",
        "--dir", config_dir,
        "-m", MODEL,
        "--format", "json",
        prompt,
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=config_dir, env=env, capture_output=True,
                               text=True, timeout=300)
        wall_clock = time.time() - t0
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        wall_clock = time.time() - t0
        stdout, stderr, rc = (e.stdout or ""), (e.stderr or "TIMEOUT"), -1

    # Extract assistant text from opencode's --format json event stream:
    # it's a sequence of newline-delimited JSON events; take the concatenation
    # of text deltas / the final message text if present, else fall back to
    # raw stdout.
    text_out = extract_text(stdout) or stdout

    ext = task["output_ext"]
    out_file = os.path.join(out_dir_abs, f"{task_id}_run{seed}.{ext}")
    with open(out_file, "w") as f:
        f.write(text_out)

    tool_calls, search_calls = count_tool_calls(stdout, query_log)
    tokens = extract_tokens(stdout)
    metrics = {
        "arm": arm,
        "task": task_id,
        "seed": seed,
        "wall_clock_sec": round(wall_clock, 2),
        "return_code": rc,
        "tool_calls_total": tool_calls,
        "search_calls": search_calls,
        "search_enabled_for_arm": search_enabled,
        "output_bytes": len(text_out),
        "model": MODEL,
        "tokens_input": tokens["input"],
        "tokens_output": tokens["output"],
        "tokens_reasoning": tokens["reasoning"],
        "tokens_cache_read": tokens["cache_read"],
        "tokens_cache_write": tokens["cache_write"],
        "tokens_total": tokens["total"],
        "cost": tokens["cost"],
    }
    metrics_file = os.path.join(out_dir_abs, f"{task_id}_run{seed}.metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    if rc != 0 or not text_out.strip():
        record_failure(run_id, arm, task_id, seed, rc, stderr[-2000:])

    return metrics, out_file


def extract_text(stdout):
    chunks = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        # opencode's JSON event stream nests message parts; be liberal here
        # since exact schema varies by version — walk for any "text" field.
        for text in find_texts(evt):
            chunks.append(text)
    return "\n".join(chunks).strip()


def find_texts(obj):
    out = []
    if isinstance(obj, dict):
        if obj.get("type") in ("text", "text-delta") and isinstance(obj.get("text"), str):
            out.append(obj["text"])
        for v in obj.values():
            out.extend(find_texts(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(find_texts(v))
    return out


def extract_tokens(stdout):
    """Sum token usage across all step_finish/step-finish events in the
    opencode --format json event stream. Each step's `part.tokens` has
    input/output/reasoning/cache{read,write} and `part.cost`."""
    totals = {"input": 0, "output": 0, "reasoning": 0,
              "cache_read": 0, "cache_write": 0, "total": 0, "cost": 0.0}
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") not in ("step_finish", "step-finish"):
            continue
        part = evt.get("part", {})
        tok = part.get("tokens", {})
        totals["input"] += tok.get("input", 0)
        totals["output"] += tok.get("output", 0)
        totals["reasoning"] += tok.get("reasoning", 0)
        totals["cache_read"] += tok.get("cache", {}).get("read", 0)
        totals["cache_write"] += tok.get("cache", {}).get("write", 0)
        totals["total"] += tok.get("total", 0)
        totals["cost"] += part.get("cost", 0) or 0
    return totals


def count_tool_calls(stdout, query_log):
    total = 0
    for line in stdout.splitlines():
        if '"type":"tool' in line or '"type": "tool' in line:
            total += 1
    search_calls = 0
    if os.path.exists(query_log):
        with open(query_log) as f:
            search_calls = sum(1 for _ in f)
    return total, search_calls


def record_failure(run_id, arm, task_id, seed, rc, stderr_tail):
    if not run_id:
        return
    failures_path = os.path.join(REPO_ROOT, "benchmarks", "results", run_id, "failures.md")
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(failures_path, "a") as f:
        f.write(
            f"\n{ts} — HARNESS RUN FAILURE: arm={arm} task={task_id} seed={seed} "
            f"return_code={rc}. Attempted: opencode run via run_harness.py. "
            f"stderr tail:\n```\n{stderr_tail}\n```\n"
        )


def log_run(run_id, arm, task_id, seed, metrics):
    if not run_id:
        return
    log_path = os.path.join(REPO_ROOT, "benchmarks", "results", run_id, "run-log.md")
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(log_path, "a") as f:
        f.write(
            f"{ts} — arm={arm} task={task_id} seed={seed} rc={metrics['return_code']} "
            f"wall_clock={metrics['wall_clock_sec']}s tool_calls={metrics['tool_calls_total']} "
            f"search_calls={metrics['search_calls']} output_bytes={metrics['output_bytes']} "
            f"tokens_in={metrics['tokens_input']} tokens_out={metrics['tokens_output']} "
            f"tokens_total={metrics['tokens_total']} cost={metrics['cost']}\n"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARM_DIRS))
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--run-id", default=os.environ.get("MOE_BENCH_RUN_ID", ""))
    args = ap.parse_args()

    metrics, out_file = run(args.arm, args.task, args.seed, args.run_id, None)
    log_run(args.run_id, args.arm, args.task, args.seed, metrics)
    print(json.dumps(metrics, indent=2))
    print("output:", out_file)
