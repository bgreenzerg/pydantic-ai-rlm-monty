# Sandbox lifecycle benchmark

Command run on the Windows development workstation on 2026-09-15:

```powershell
.\.venv\Scripts\python.exe benchmarks\sandbox_lifecycle.py --iterations 100 --context-bytes 1048576
```

Result after a warm-up run:

| Measure | Result |
|---|---:|
| Iterations | 100 |
| Context per run | 1,048,576 bytes |
| Lifecycle latency p50 | 28.860 ms |
| Lifecycle latency p95 | 31.350 ms |
| Lifecycle latency max | 31.919 ms |
| Parent RSS delta | 81,920 bytes |
| Leaked child processes | 0 |

Lifecycle latency includes worker creation, context transfer, one code execution
and worker teardown, plus a fresh SHA-256 verification of the 25 MB native worker
before each run. This is a local engineering measurement, not a capacity
claim. Repeat load and soak testing on the production OS, hardware, concurrency,
context distribution and model-provider path.
