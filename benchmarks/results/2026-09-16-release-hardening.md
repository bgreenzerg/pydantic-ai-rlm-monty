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
| Active sandbox lifecycle | p50 | 43.702 ms | 39.317 ms | −10.03% |
| Active sandbox lifecycle | p95 | 47.172 ms | 46.872 ms | −0.64% |
| No-tool agent run | p50 | 46.629 ms | 3.941 ms | −91.55% |
| No-tool agent run | total, 40 runs | 1,886.020 ms | 242.958 ms | −87.12% |
| No-tool agent run | peak Monty workers | 1 | 0 | −100% |

The active path includes worker startup, 1 MiB context transfer, one execution
and teardown. It did not regress. Lazy worker acquisition removes all Monty
startup cost and worker RSS from runs whose model never calls `execute_code`.
No worker process leaked in either scenario or revision.

Peak parent RSS deltas were approximately 2.58 MB versus 3.11 MB for the active
path and 4.05 MB versus 1.97 MB for the no-tool path. RSS is sampled and noisy,
so it is reported as supporting evidence rather than used as a pass/fail gate.
The machine-readable local report remains under the ignored `benchmark-results/`
directory.

This final comparison used candidate commit `bd454b6`; the candidate worktree
was clean when measured.

## Final live large-case verification

Both deterministic large cases passed against OpenRouter model
`openai/gpt-5.6-luna` with read-back from the local MLflow experiment
`pydantic-ai-rlm-large-benchmark`:

| Case | Time | Code calls | Nested calls | Peak worker RSS | MLflow trace |
|---|---:|---:|---:|---:|---|
| Customer feedback, 5,000 records | 61.931 s | 8 | 1/1 completed | 23.081 MB | `tr-da4d48119236ca3bfeb35b14e09c4caf` |
| Accounting, 3,682 records | 56.573 s | 8 | 0 | 21.082 MB | `tr-59761fd16cea0d87cd4893fb16fc512c` |

Both runs passed fixture-integrity, domain-quality, bounded-trajectory,
credential-absence, trace-integrity and worker-cleanup checks. No Monty worker
remained after either run.
