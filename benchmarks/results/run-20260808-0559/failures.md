# Failures — run-20260808-0559

2026-08-08T06:00:43Z — Not a run failure, but a setup gap: rocm-smi, rocminfo, amd-smi, and torch are not installed on this VM despite the MI300X GPU being physically present (confirmed via lspci). Since the benchmarked agent (OpenCode + DeepSeek V4 Flash via API) does not require local GPU inference, this is not blocking, but is recorded here since it affects the completeness of GPU utilization telemetry for cost metrics (H3/task4 GPU utilization row) — that row will be marked N/A with this note rather than fabricated.
2026-08-08T06:00:43Z — pip3 not found on VM; will install via apt or ensurepip before proceeding.

2026-08-08T06:38:54Z — CONTAMINATION FAILURE (caught during headroom check, fixed before full wave): arm A1 task2 seed2 first attempt burned 44 tool calls / 524,194 tokens over 300s and timed out without producing output. Root cause: opencode configs disabled bash/edit/write/webfetch but left the built-in read/grep/glob/list tools enabled, letting the model browse the harness repo (including score_architecture.py, skill worked examples, and a prior run's own output) mid-run, explicitly to "produce a coherent, high-scoring config" per its own stated reasoning. This is the skill-rubric-circularity threat materializing through an unintended tool-access channel. RESOLUTION: added read/grep/glob/list:false to all four benchmarks/harness/configs/{A0,A1,A2,A3}/opencode.jsonc, verified via a probe run showing zero tool-call events post-fix. All headroom-check data re-generated after the fix; original contaminated task2_run2 files were deleted, not scored. Full wave must re-verify this via spot-checks on run-log.md tool-call counts.

2026-08-08T07:31:51Z — WAVE DRIVER: arm=A3 task=task2 seed=1 failed (rc/timeout). Retrying once.
```
        ~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/moe-agentic-development/benchmarks/harness/run_harness.py", line 109, in run
    text_out = extract_text(stdout) or stdout
               ~~~~~~~~~~~~^^^^^^^^
  File "/root/moe-agentic-development/benchmarks/harness/run_harness.py", line 151, in extract_text
    if not line or not line.startswith("{"):
                       ~~~~~~~~~~~~~~~^^^^^
TypeError: startswith first arg must be bytes or a tuple of bytes, not str

```
2026-08-08T07:33:15Z — RETRY SUCCEEDED for arm=A3 task=task2 seed=1.

2026-08-08T07:46:21Z — WAVE DRIVER: arm=A1 task=task3 seed=5 failed (rc/timeout). Retrying once.
```
        ~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/moe-agentic-development/benchmarks/harness/run_harness.py", line 109, in run
    text_out = extract_text(stdout) or stdout
               ~~~~~~~~~~~~^^^^^^^^
  File "/root/moe-agentic-development/benchmarks/harness/run_harness.py", line 151, in extract_text
    if not line or not line.startswith("{"):
                       ~~~~~~~~~~~~~~~^^^^^
TypeError: startswith first arg must be bytes or a tuple of bytes, not str

```
2026-08-08T07:46:40Z — RETRY SUCCEEDED for arm=A1 task=task3 seed=5.

2026-08-08T08:10:05Z — WAVE DRIVER: arm=A3 task=task4 seed=2 failed (rc/timeout). Retrying once.
```
        ~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/moe-agentic-development/benchmarks/harness/run_harness.py", line 109, in run
    text_out = extract_text(stdout) or stdout
               ~~~~~~~~~~~~^^^^^^^^
  File "/root/moe-agentic-development/benchmarks/harness/run_harness.py", line 151, in extract_text
    if not line or not line.startswith("{"):
                       ~~~~~~~~~~~~~~~^^^^^
TypeError: startswith first arg must be bytes or a tuple of bytes, not str

```
2026-08-08T08:10:54Z — RETRY SUCCEEDED for arm=A3 task=task4 seed=2.
