# Parallel verification — 2026-09-18

Tested working tree based on b9176ca, Windows, Python 3.14.3. No package
implementation changes. Reproduce with:

```powershell
.venv\Scripts\python.exe benchmarks/parallel_verification.py
```

Real Monty subprocesses and shared Pydantic AI agent, deterministic FunctionModel
instead of network LLM calls. Each run has 1 MiB of text context, a unique tenant
marker, two code calls, and a model wait between calls. Checks ensure state
persists within a run and returned markers do not cross tenants.

| Scenario | Result |
| --- | --- |
| 32 concurrent runs, one host process, 0.2 s model wait | All passed in 2.848 s; peak 4 held sessions |
| Fifth session while four slots occupied, default checkout timeout | TimeoutError after 10.015 s |
| Cancel another session waiting for capacity | Existing sessions unaffected; all four slots reusable |
| Four host processes, four runs each, 2 s model wait | All passed; observed 16 simultaneous Monty workers |
| Per-host four-run durations | 2.193, 2.193, 2.173, 2.188 s |
| Sampled combined Monty worker peak RSS | 167,632,896 bytes (167.63 MB; excludes Python hosts) |
| Remaining child processes | 0 |

Also: 41 existing REPL, toolset and agent integration tests passed in 2.51 s.
These cover cancellation and fatal execution failures, but do not constitute a
long-duration load test or simultaneous injected worker-crash stress test.
The timings include deliberately injected waits and are not live LLM throughput.
Workers being alive simultaneously is not a measurement of CPU parallel speedup.

## Limits in current implementation

- `repl.py` uses a hard-coded process-wide `threading.BoundedSemaphore(4)`.
  Async tasks and threads within the same process share those four slots.
  Separately started Python processes each have their own four-slot limit.
- A run acquires capacity lazily on the first code call and holds it until run
  teardown, including subsequent main-model and nested-model waits.
- Extra sessions poll every 10 ms until `checkout_timeout`, default 10 seconds,
  configurable up to 300 seconds. Expiration raises TimeoutError and aborts the
  affected run; there is no automatic retry or durable queue. FIFO fairness is
  not guaranteed. The number of waiting runs/context objects is not bounded.
- `execute_code` is sequential within each agent run. Parallelism is across
  independently owned runs, not concurrent calls on one REPL instance.
- Memory budgets are per sandbox; default Monty memory limit is 256 MiB.
  This is not a cap on combined Python host/worker RSS. More host processes
  multiply capacity and resource consumption; there is no machine-wide limit.
- Use independently started/spawned application workers. This test used fresh
  subprocesses, not POSIX fork of an application holding live Monty resources.

## Additional confirmed defect

This minimal structured context fails before sandbox startup:

```python
RLMDependencies(context={"a": "one", "b": "two"})
```

`dependencies.py:261` builds a generator referring to the loop variable `item`.
When the outer traversal assigns the next child to `item`, subsequent dictionary
lookups use the child instead of the original dictionary. The reproduction raises
`TypeError: string indices must be integers, not 'str'`. Text contexts avoid this
path. This needs correction before relying on structured-context support; the
current passing tests do not cover the minimal reproduction.
