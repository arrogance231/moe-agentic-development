#!/usr/bin/env python3
"""Driver for the full benchmark wave: 6 tasks x 4 arms x 5 seeds = 120 runs.

Runs sequentially (each opencode invocation already takes 80-300s and is
itself the unit of GPU/API cost; parallelizing would just race the same
API key). Appends to run-log.md as each run completes (run_harness.py does
this itself). Commits to benchmark-harness branch after each task's 20 runs.
"""
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.expanduser("~/moe-agentic-development")
HARNESS = os.path.join(REPO_ROOT, "benchmarks", "harness", "run_harness.py")
RUN_ID = "run-20260808-0559"
TASKS = ["task1", "task2", "task3", "task4", "task5", "task6"]
ARMS = ["A0", "A1", "A2", "A3"]
SEEDS = [1, 2, 3, 4, 5]

PROGRESS_FILE = os.path.join(REPO_ROOT, "benchmarks", "results", RUN_ID, "wave-progress.json")


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)


def git(*args):
    subprocess.run(["git", "-C", REPO_ROOT] + list(args), check=False)


def main():
    progress = load_progress()
    done_set = set(tuple(x) for x in progress["completed"])

    for task in TASKS:
        task_had_new = False
        for arm in ARMS:
            for seed in SEEDS:
                key = [arm, task, seed]
                if tuple(key) in done_set:
                    continue
                print(f"=== RUN arm={arm} task={task} seed={seed} ===", flush=True)
                env = dict(os.environ)
                env["MOE_BENCH_RUN_ID"] = RUN_ID
                t0 = time.time()
                try:
                    proc = subprocess.run(
                        [sys.executable, HARNESS, "--arm", arm, "--task", task,
                         "--seed", str(seed), "--run-id", RUN_ID],
                        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                        timeout=340,
                    )
                    ok = proc.returncode == 0
                    tail = (proc.stdout[-500:] + proc.stderr[-500:])
                except subprocess.TimeoutExpired:
                    ok = False
                    tail = "DRIVER TIMEOUT (340s)"
                dt = time.time() - t0
                print(f"    -> ok={ok} dt={dt:.1f}s", flush=True)

                if ok:
                    progress["completed"].append(key)
                    done_set.add(tuple(key))
                    task_had_new = True
                else:
                    progress["failed"].append({"arm": arm, "task": task, "seed": seed, "tail": tail})
                    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    fail_path = os.path.join(REPO_ROOT, "benchmarks", "results", RUN_ID, "failures.md")
                    with open(fail_path, "a") as f:
                        f.write(
                            f"\n{ts} — WAVE DRIVER: arm={arm} task={task} seed={seed} failed "
                            f"(rc/timeout). Retrying once.\n```\n{tail}\n```\n"
                        )
                    # retry once
                    try:
                        proc = subprocess.run(
                            [sys.executable, HARNESS, "--arm", arm, "--task", task,
                             "--seed", str(seed), "--run-id", RUN_ID],
                            cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                            timeout=340,
                        )
                        ok2 = proc.returncode == 0
                    except subprocess.TimeoutExpired:
                        ok2 = False
                    with open(fail_path, "a") as f:
                        ts2 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        if ok2:
                            f.write(f"{ts2} — RETRY SUCCEEDED for arm={arm} task={task} seed={seed}.\n")
                            progress["completed"].append(key)
                            done_set.add(tuple(key))
                            task_had_new = True
                        else:
                            f.write(
                                f"{ts2} — RETRY FAILED for arm={arm} task={task} seed={seed}. "
                                f"EXCLUDING from analysis.\n"
                            )
                save_progress(progress)

        if task_had_new:
            git("add", "benchmarks/")
            git("commit", "-m", f"Wave progress: completed runs for {task}")
            print(f"=== committed progress for {task} ===", flush=True)

    print("=== WAVE COMPLETE ===", flush=True)
    print(f"completed={len(progress['completed'])} failed_excluded={len(progress['failed'])}", flush=True)


if __name__ == "__main__":
    main()
