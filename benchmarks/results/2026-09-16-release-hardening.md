# Release-hardening performance comparison — 2026-09-16

The reusable comparison script was run against pre-hardening commit `d5de13d`
and the 0.2.1 candidate working tree on the same Windows host, Python 3.14.3
interpreter and dependency environment:

```powershell
.\.venv\Scripts\python.exe benchmarks\compare_release_performance.py `
  --baseline-ref d5de13d --iterations 40 --context-bytes 1048576 `
  --output benchmark-results\release-performance-0.2.1.json
```

| Scenario | Measure | 0.2.0 baseline | 0.2.1 candidate | Change |
|---|---|---:|---:|---:|
| Active sandbox lifecycle | p50 | 47.341 ms | 46.723 ms | −1.31% |
| Active sandbox lifecycle | p95 | 57.935 ms | 58.178 ms | +0.42% |
| No-tool agent run | p50 | 47.780 ms | 5.707 ms | −88.06% |
| No-tool agent run | total, 40 runs | 2,019.592 ms | 311.107 ms | −84.60% |
| No-tool agent run | peak Monty workers | 1 | 0 | −100% |

The active path includes worker startup, 1 MiB context transfer, one execution
and teardown. It did not regress. Lazy worker acquisition removes all Monty
startup cost and worker RSS from runs whose model never calls `execute_code`.
No worker process leaked in either scenario or revision.

Peak parent RSS deltas were approximately 2.33 MB versus 2.18 MB for the active
path and 4.57 MB versus 1.95 MB for the no-tool path. RSS is sampled and noisy,
so it is reported as supporting evidence rather than used as a pass/fail gate.
The machine-readable local report remains under the ignored `benchmark-results/`
directory.
