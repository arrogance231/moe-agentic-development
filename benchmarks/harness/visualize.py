#!/usr/bin/env python3
import json
import os
import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/moe-agentic-development")
RESULTS = os.path.join(ROOT, "benchmarks/results/run-20260808-0559")
VIZ = os.path.join(RESULTS, "visualizations")
os.makedirs(VIZ, exist_ok=True)

ARMS = ["A0", "A1", "A2", "A3"]
TASKS = [f"task{i}" for i in range(1, 7)]
COLORS = {"A0": "#94a3b8", "A1": "#60a5fa", "A2": "#fbbf24", "A3": "#34d399"}

with open(os.path.join(RESULTS, "scored_dataset.json")) as f:
    rows = json.load(f)
by_key = {(r["arm"], r["task"], r["seed"]): r for r in rows}


def primary_metric(row):
    task, score = row["task"], row["score"]
    if score is None:
        return None
    if task == "task1":
        return score.get("total")
    if task == "task2":
        return score.get("successful_launch")
    if task == "task3":
        return score.get("accuracy")
    if task == "task4":
        return score.get("throughput_delta_frac")
    if task == "task5":
        return score.get("internal_consistency")
    if task == "task6":
        return (score.get("architecture") or {}).get("total")


# 1. per-arm score distributions per task (grid of 6 subplots)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, task in zip(axes.flat, TASKS):
    means, sds = [], []
    for arm in ARMS:
        vals = [primary_metric(by_key[(arm, task, s)]) for s in range(1, 6)
                if by_key.get((arm, task, s)) and primary_metric(by_key[(arm, task, s)]) is not None]
        m = st.mean(vals) if vals else 0
        sd = st.stdev(vals) if len(vals) > 1 else 0
        means.append(m)
        sds.append(sd)
    ax.bar(ARMS, means, yerr=sds, capsize=4, color=[COLORS[a] for a in ARMS])
    ax.set_title(task)
    ax.set_ylabel("score")
fig.suptitle("Per-arm score distributions per task (mean ± SD, n=5)")
fig.tight_layout()
fig.savefig(os.path.join(VIZ, "score_distributions.png"), dpi=130)
plt.close(fig)

# 2. cost by arm (tokens, tool calls, wall-clock) aggregated across tasks
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
metrics_keys = [("tokens_total", "mean tokens/run"), ("tool_calls_total", "mean tool calls/run"),
                ("wall_clock_sec", "mean wall-clock (s)/run")]
for ax, (key, label) in zip(axes, metrics_keys):
    means = []
    for arm in ARMS:
        vals = []
        for task in TASKS:
            for s in range(1, 6):
                r = by_key.get((arm, task, s))
                if r and r.get("metrics"):
                    vals.append(r["metrics"].get(key, 0))
        means.append(st.mean(vals) if vals else 0)
    ax.bar(ARMS, means, color=[COLORS[a] for a in ARMS])
    ax.set_title(label)
fig.suptitle("Cost by arm (aggregated across all 6 tasks, n=30/arm)")
fig.tight_layout()
fig.savefig(os.path.join(VIZ, "cost_by_arm.png"), dpi=130)
plt.close(fig)

# 3. score variance (SD) by arm, per task
fig, ax = plt.subplots(figsize=(10, 5))
width = 0.2
x = range(len(TASKS))
for i, arm in enumerate(ARMS):
    sds = []
    for task in TASKS:
        vals = [primary_metric(by_key[(arm, task, s)]) for s in range(1, 6)
                if by_key.get((arm, task, s)) and primary_metric(by_key[(arm, task, s)]) is not None]
        sds.append(st.stdev(vals) if len(vals) > 1 else 0)
    ax.bar([xi + i * width for xi in x], sds, width=width, label=arm, color=COLORS[arm])
ax.set_xticks([xi + 1.5 * width for xi in x])
ax.set_xticklabels(TASKS)
ax.set_ylabel("score SD (n=5)")
ax.set_title("Score variance (SD) by arm, per task — H2 endpoint")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(VIZ, "score_variance_by_arm.png"), dpi=130)
plt.close(fig)

print("wrote 3 PNGs to", VIZ)
