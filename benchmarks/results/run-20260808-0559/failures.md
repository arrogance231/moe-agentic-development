# Failures — run-20260808-0559

2026-08-08T06:00:43Z — Not a run failure, but a setup gap: rocm-smi, rocminfo, amd-smi, and torch are not installed on this VM despite the MI300X GPU being physically present (confirmed via lspci). Since the benchmarked agent (OpenCode + DeepSeek V4 Flash via API) does not require local GPU inference, this is not blocking, but is recorded here since it affects the completeness of GPU utilization telemetry for cost metrics (H3/task4 GPU utilization row) — that row will be marked N/A with this note rather than fabricated.
2026-08-08T06:00:43Z — pip3 not found on VM; will install via apt or ensurepip before proceeding.
