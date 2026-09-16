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
| Active sandbox lifecycle | p50 | 46.665 ms | 46.405 ms | −0.56% |
| Active sandbox lifecycle | p95 | 48.991 ms | 48.112 ms | −1.79% |
| No-tool agent run | p50 | 47.526 ms | 7.306 ms | −84.63% |
| No-tool agent run | total, 40 runs | 1,964.782 ms | 314.345 ms | −84.00% |
| No-tool agent run | peak Monty workers | 1 | 0 | −100% |

The active path includes worker startup, 1 MiB context transfer, one execution
and teardown. It did not regress. Lazy worker acquisition removes all Monty
startup cost and worker RSS from runs whose model never calls `execute_code`.
No worker process leaked in either scenario or revision.

Peak parent RSS deltas were approximately 2.76 MB versus 2.22 MB for the active
path and 4.58 MB versus 1.98 MB for the no-tool path. RSS is sampled and noisy,
so it is reported as supporting evidence rather than used as a pass/fail gate.
The machine-readable local report remains under the ignored `benchmark-results/`
directory.
