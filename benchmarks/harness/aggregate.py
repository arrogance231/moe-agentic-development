#!/usr/bin/env python3
"""Aggregate scored_dataset.json into results-<task>.md and summary.md."""
import json
import os
import statistics as st
from scipy import stats as sps

ROOT = os.path.expanduser("~/moe-agentic-development")
RESULTS = os.path.join(ROOT, "benchmarks/results/run-20260808-0559")
ARMS = ["A0", "A1", "A2", "A3"]
TASKS = [f"task{i}" for i in range(1, 7)]

with open(os.path.join(RESULTS, "scored_dataset.json")) as f:
    rows = json.load(f)

by_key = {}
for r in rows:
    by_key[(r["arm"], r["task"], r["seed"])] = r


def primary_metric(row):
    task, score = row["task"], row["score"]
    if score is None:
        return None
    if task == "task1":
        return score.get("total") if score else None
    if task == "task2":
        return score.get("successful_launch")
    if task == "task3":
        return score.get("accuracy")
    if task == "task4":
        return score.get("throughput_delta_frac")
    if task == "task5":
        return score.get("internal_consistency")
    if task == "task6":
        arch = score.get("architecture") or {}
        return arch.get("total")


def cohens_d(a, b):
    diffs = [x - y for x, y in zip(a, b)]
    if len(diffs) < 2 or st.stdev(diffs) == 0:
        return None
    return st.mean(diffs) / st.stdev(diffs)


def paired_test(a, b):
    diffs = [x - y for x, y in zip(a, b)]
    if all(d == diffs[0] for d in diffs):
        # constant diff / no variance -> report descriptive, skip test
        return None, None
    try:
        t, p = sps.ttest_rel(a, b)
        return float(t), float(p)
    except Exception:
        return None, None


def holm_correct(pvals_labeled):
    # pvals_labeled: list of (label, p) with p not None
    items = [(l, p) for l, p in pvals_labeled if p is not None]
    items.sort(key=lambda x: x[1])
    m = len(items)
    out = {}
    max_adj = 0
    for i, (l, p) in enumerate(items):
        adj = p * (m - i)
        adj = min(adj, 1.0)
        max_adj = max(max_adj, adj)
        out[l] = max(adj, max_adj) if i > 0 else adj
    return out


agg = {}
for task in TASKS:
    agg[task] = {}
    for arm in ARMS:
        vals = []
        for seed in range(1, 6):
            r = by_key.get((arm, task, seed))
            if r:
                v = primary_metric(r)
                if v is not None:
                    vals.append(v)
        agg[task][arm] = vals

# cost metrics
cost = {}
for task in TASKS:
    cost[task] = {}
    for arm in ARMS:
        toks, calls, wc = [], [], []
        for seed in range(1, 6):
            r = by_key.get((arm, task, seed))
            if r and r.get("metrics"):
                m = r["metrics"]
                toks.append(m.get("tokens_total", 0))
                calls.append(m.get("tool_calls_total", 0))
                wc.append(m.get("wall_clock_sec", 0))
        cost[task][arm] = {"tokens": toks, "calls": calls, "wc": wc}

table_meta = {
    "task1": ("correctness (total /25)", "0-25"),
    "task2": ("successful launch", "0/1"),
    "task3": ("accuracy (0-1)", "0-1"),
    "task4": ("throughput delta (frac)", "ratio"),
    "task5": ("internal consistency", "0/1"),
    "task6": ("constraint satisfaction + arch total", "0/1, 0-25"),
}


def mean_sd(vals):
    if not vals:
        return None, None
    m = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return m, sd


# ---- per-task results files ----
for task in TASKS:
    metric_name, scale = table_meta[task]
    lines = [f"# Results — {task} ({RESULTS.split('/')[-1]})\n"]
    lines.append(f"Metric: {metric_name} ({scale}). n=5 seeds per arm.\n")
    lines.append("| Arm | n | mean | SD | values |")
    lines.append("|---|---|---|---|---|")
    for arm in ARMS:
        vals = agg[task][arm]
        m, sd = mean_sd(vals)
        lines.append(f"| {arm} | {len(vals)} | {m if m is None else round(m,3)} | "
                      f"{sd if sd is None else round(sd,3)} | {vals} |")
    a1 = agg[task]["A1"]
    a3 = agg[task]["A3"]
    d = cohens_d(a3, a1) if len(a1) == len(a3) == 5 else None
    t, p = paired_test(a3, a1) if len(a1) == len(a3) == 5 else (None, None)
    lines.append("")
    lines.append(f"**A3 vs A1 (primary comparison for this task):** "
                  f"Cohen's d = {round(d,3) if d is not None else 'n/a (zero variance in paired diffs)'}, "
                  f"paired t-test p = {round(p,4) if p is not None else 'n/a'}")
    lines.append("")
    lines.append("### Cost (this task)")
    lines.append("| Arm | mean tokens | mean tool calls | mean wall-clock (s) |")
    lines.append("|---|---|---|---|")
    for arm in ARMS:
        c = cost[task][arm]
        mt, _ = mean_sd(c["tokens"])
        mc, _ = mean_sd(c["calls"])
        mw, _ = mean_sd(c["wc"])
        lines.append(f"| {arm} | {round(mt,1) if mt else mt} | {round(mc,2) if mc else mc} | "
                      f"{round(mw,1) if mw else mw} |")
    if task == "task1":
        lines.append("")
        lines.append("**Ceiling-effect caveat:** the A1-only headroom check (n=2, "
                      "see headroom-check.md) found scores of 21/25 and 25/25 — already "
                      "near the rubric maximum. This task has reduced statistical power "
                      "to detect an A3-vs-A1 effect and results here should be read with "
                      "that in mind, per methodology.md's pre-registered ceiling-effect "
                      "handling.")
    with open(os.path.join(RESULTS, f"results-{task}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

print("wrote results-<task>.md for all 6 tasks")

# ---- summary.md ----
lines = []
lines.append(f"# Summary — run-20260808-0559\n")
lines.append("Full four-arm (A0/A1/A2/A3) x six-task benchmark, n=5 seeds per "
              "arm per task, 120/120 runs completed (5 transient errors, all "
              "resolved via retry, zero permanent exclusions). Model under test: "
              "OpenCode CLI driving `opencode/deepseek-v4-flash-free` via the "
              "OpenCode Zen API. Retrieval backend: keyless DuckDuckGo Lite HTML "
              "scrape (no paid search API available) — logged as a validity "
              "limitation (rate limits, less reliable structured results than a "
              "paid backend such as Tavily/Brave).\n")

lines.append("## Sample size caveat\n")
lines.append("n=5 paired runs per arm per task is the BENCHMARK.md-specified "
              "minimum. All effect sizes and p-values below should be read as "
              "low-powered point estimates, not confirmatory results. This "
              "matches the design's stated intent to power variance (H2) and "
              "cost (H3) endpoints fully while treating quality endpoints (H1) "
              "as directional at this n.\n")

# retrieval-only gain FIRST
lines.append("## Retrieval-only gain (A1 − A0) — reported before any skill effect\n")
lines.append("Per BENCHMARK.md's design intent, this is the number a skeptical "
              "reader looks for first: how much of any observed effect is "
              "available from search alone, with no skills involved.\n")
lines.append("| Task | Metric | A0 mean | A1 mean | A1−A0 | Cohen's d |")
lines.append("|---|---|---|---|---|---|")
secondary_pvals = []
for task in TASKS:
    metric_name, _ = table_meta[task]
    a0 = agg[task]["A0"]
    a1 = agg[task]["A1"]
    m0, _ = mean_sd(a0)
    m1, _ = mean_sd(a1)
    diff = (m1 - m0) if (m0 is not None and m1 is not None) else None
    d = cohens_d(a1, a0) if len(a0) == len(a1) == 5 else None
    lines.append(f"| {task} | {metric_name} | {round(m0,3) if m0 is not None else m0} | "
                  f"{round(m1,3) if m1 is not None else m1} | "
                  f"{round(diff,3) if diff is not None else diff} | "
                  f"{round(d,3) if d is not None else 'n/a'} |")
    _, p = paired_test(a1, a0) if len(a0) == len(a1) == 5 else (None, None)
    secondary_pvals.append((f"{task}:A1-A0", p))

lines.append("")
lines.append("## Primary endpoint: A3 vs A1 (pre-registered)\n")
lines.append("| Task | Metric | A1 mean | A3 mean | A3−A1 | Cohen's d | paired t p |")
lines.append("|---|---|---|---|---|---|---|")
primary_pvals = []
for task in TASKS:
    metric_name, _ = table_meta[task]
    a1 = agg[task]["A1"]
    a3 = agg[task]["A3"]
    m1, _ = mean_sd(a1)
    m3, _ = mean_sd(a3)
    diff = (m3 - m1) if (m1 is not None and m3 is not None) else None
    d = cohens_d(a3, a1) if len(a1) == len(a3) == 5 else None
    t, p = paired_test(a3, a1) if len(a1) == len(a3) == 5 else (None, None)
    lines.append(f"| {task} | {metric_name} | {round(m1,3) if m1 is not None else m1} | "
                  f"{round(m3,3) if m3 is not None else m3} | "
                  f"{round(diff,3) if diff is not None else diff} | "
                  f"{round(d,3) if d is not None else 'n/a'} | "
                  f"{round(p,4) if p is not None else 'n/a'} |")
    primary_pvals.append((task, p))

lines.append("")
lines.append("**Task1 ceiling-effect caveat carried forward:** the A1-only "
              "headroom check found task1 scores near the rubric max (21/25, "
              "25/25 at n=2). The full-wave task1 numbers above should be read "
              "with reduced power to detect an A3-vs-A1 difference for that "
              "reason, consistent with methodology.md's pre-registered handling "
              "of ceiling effects (report the saturation, do not force a null "
              "result to look like evidence of no effect).\n")

lines.append("## Secondary comparisons (exploratory, Holm-corrected)\n")
lines.append("Per methodology.md, all comparisons other than A3 vs A1 are "
              "secondary and reported as exploratory, not confirmatory. Holm "
              "correction applied across the full secondary family below.\n")

secondary_pvals2 = []
extra_rows = []
for task in TASKS:
    a0 = agg[task]["A0"]; a1 = agg[task]["A1"]; a2 = agg[task]["A2"]; a3 = agg[task]["A3"]
    for label, x, y in [
        (f"{task}:A2-A0", a2, a0),
        (f"{task}:A2vA1", a2, a1),
        (f"{task}:A3-A2", a3, a2),
    ]:
        if len(x) == len(y) == 5:
            _, p = paired_test(x, y)
        else:
            p = None
        secondary_pvals2.append((label, p))
        mx, _ = mean_sd(x); my, _ = mean_sd(y)
        diff = (mx - my) if (mx is not None and my is not None) else None
        extra_rows.append((label, diff, p))

all_secondary = secondary_pvals + secondary_pvals2
holm = holm_correct(all_secondary)

lines.append("| Comparison | mean diff | raw p | Holm-adjusted p |")
lines.append("|---|---|---|---|")
for task in TASKS:
    a0 = agg[task]["A0"]; a1 = agg[task]["A1"]
    m0, _ = mean_sd(a0); m1, _ = mean_sd(a1)
    diff = (m1 - m0) if (m0 is not None and m1 is not None) else None
    label = f"{task}:A1-A0"
    p = dict(secondary_pvals).get(label)
    adj = holm.get(label)
    lines.append(f"| {label} | {round(diff,3) if diff is not None else diff} | "
                  f"{round(p,4) if p is not None else 'n/a'} | "
                  f"{round(adj,4) if adj is not None else 'n/a'} |")
for label, diff, p in extra_rows:
    adj = holm.get(label)
    lines.append(f"| {label} | {round(diff,3) if diff is not None else diff} | "
                  f"{round(p,4) if p is not None else 'n/a'} | "
                  f"{round(adj,4) if adj is not None else 'n/a'} |")

lines.append("")
lines.append("## Reliability endpoints (H2, H5)\n")
lines.append("| Task | SD(A0) | SD(A1) | SD(A2) | SD(A3) |")
lines.append("|---|---|---|---|---|")
for task in TASKS:
    sds = []
    for arm in ARMS:
        _, sd = mean_sd(agg[task][arm])
        sds.append(round(sd,3) if sd is not None else None)
    lines.append(f"| {task} | {sds[0]} | {sds[1]} | {sds[2]} | {sds[3]} |")

lines.append("")
lines.append("## Cost endpoints (H3) — aggregated across all 6 tasks\n")
lines.append("| Arm | mean tokens/run | mean tool calls/run | mean wall-clock (s)/run |")
lines.append("|---|---|---|---|")
for arm in ARMS:
    toks, calls, wc = [], [], []
    for task in TASKS:
        toks += cost[task][arm]["tokens"]
        calls += cost[task][arm]["calls"]
        wc += cost[task][arm]["wc"]
    mt, _ = mean_sd(toks); mc, _ = mean_sd(calls); mw, _ = mean_sd(wc)
    lines.append(f"| {arm} | {round(mt,1) if mt else mt} | {round(mc,2) if mc else mc} | "
                  f"{round(mw,1) if mw else mw} |")

lines.append("")
lines.append("## What is and isn't supported by this data\n")
lines.append(
"""- Retrieval-only gain (A1−A0) is reported above per-task, before any skill
  effect; it is the honest floor of "how much does search alone buy you."
- The pre-registered primary endpoint (A3 vs A1) is reported per task with
  Cohen's d and a paired t-test p-value at n=5 — these are point estimates
  from a small sample and are not strong enough to claim statistical
  significance on their own for any single task; treat directionality
  (sign and rough magnitude of A3−A1) as the informative signal at this n,
  not the p-value threshold.
- Task1 has a documented ceiling effect (near-max scores in the headroom
  check); any near-zero A3−A1 difference there is consistent with "no
  headroom to detect an effect," not necessarily "skills add nothing."
- Task2/task4 numeric fields (memory efficiency, throughput, GPU utilization)
  are self-reported by the model under test — no real training job executes
  in this benchmark. These are internal-consistency/plausibility signals,
  not measured ground truth. This is a real limitation of the benchmark as
  currently specified in BENCHMARK.md (which does not call for an actual
  training run), stated here rather than treated as a hard result.
- Secondary comparisons (A2−A0, A2 vs A1, A3−A2) are exploratory and
  Holm-corrected; none should be read as confirmatory on their own.
- Retrieval used a keyless, unpaid backend (DuckDuckGo Lite scrape) rather
  than the paid API BENCHMARK.md's design implicitly assumes — a stated
  limitation on data quality/reliability for A1/A3, not a silent gap.
- GPU utilization telemetry (rocm-smi/rocminfo/amd-smi) was unavailable on
  this VM despite a physical MI300X being present (see environment.md);
  this does not affect the benchmark itself since the model under test runs
  via API, not local GPU inference, but it means no independently measured
  GPU utilization numbers exist for any arm — only the model's own claims
  in task4 outputs.
""")

with open(os.path.join(RESULTS, "summary.md"), "w") as f:
    f.write("\n".join(lines) + "\n")

print("wrote summary.md")
