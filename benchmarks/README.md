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

## Reproducible live synthetic suite

`live_synthetic_suite.py` runs five deterministic synthetic workloads through
the real main model, OpenRouter, and Monty sandbox:

1. needle extraction from 2,000 log lines;
2. exact aggregation of 700 transactions;
3. persistent variables across three sandbox executions;
4. semantic delegation through `llm_query`;
5. cross-run isolation, host-environment denial, and prompt-injection resistance.

The script derives the expected answers independently, checks every final answer,
counts model and tool calls, samples parent/worker RSS, checks worker teardown,
and prints a machine-readable JSON report. It loads the ignored `.env` file and
requires `OPENROUTER_API_KEY` plus `ASSISTANT_MODEL`.

```powershell
.\.venv\Scripts\python.exe benchmarks\live_synthetic_suite.py `
  --output benchmark-results\live-suite.json
```

The provider-reported token and cost totals currently cover main-agent requests
only. Nested `llm_query` calls are counted, but their token usage and cost are not
aggregated by the current implementation. Input data is synthetic, and credential
values are checked for absence from captured tool output and final answers.

## Live OpenRouter integration

`scripts/live_openrouter_smoke.py` completed successfully on 2026-09-15 using
the locally configured `openai/gpt-5.6-luna` model through OpenRouter and purely
synthetic input data.

| Measure | Result |
|---|---:|
| End-to-end duration | 17.186 seconds |
| Main-model requests | 6 |
| Monty `execute_code` calls | 5 |
| Completed nested `llm_query` calls | 2 |
| Input tokens | 10,453 |
| Output tokens | 529 |
| Provider-reported cost | $0.00106100 |
| Verification checks | 5/5 passed |

The checks prove that the main model used the RLM tool repeatedly, both nested
sub-model callbacks started and completed, and the final response contained the
control identifier extracted from context plus the required sub-model marker.
Content logging remained disabled throughout the run.
