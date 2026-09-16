# Sandbox lifecycle benchmark

## Release-to-release comparison

`compare_release_performance.py` checks the current working tree against a Git
baseline using the same Python interpreter, dependency environment and machine.
It benchmarks both active sandbox lifecycle and agent runs that do not call the
sandbox, then enforces worker-leak, active-latency and lazy-worker gates.

```powershell
.\.venv\Scripts\python.exe benchmarks\compare_release_performance.py `
  --baseline-ref d5de13d --iterations 40 --context-bytes 1048576 `
  --output benchmark-results\release-performance.json
```

The baseline source is read through `git archive` into a temporary directory;
the repository and baseline commit are not modified. RSS values are sampled and
reported, but are deliberately not a pass/fail gate because short-process RSS is
noisy. The JSON report includes exact commit IDs and whether the candidate tree
contained uncommitted changes.

## Single-version lifecycle measurement

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
and prints a machine-readable JSON report. It also requires MLflow tracing: each
case becomes a root trace with nested Pydantic AI agent, model, and tool spans.
Trace IDs and the shared suite ID are included in the JSON report. The benchmark
fails if all five root traces cannot be read back from MLflow.

Install the two optional dependency groups before running it:

```powershell
.\.venv\Scripts\uv.exe sync --extra openrouter --extra observability
```

The script loads the ignored `.env` file and requires `OPENROUTER_API_KEY` plus
`ASSISTANT_MODEL`. MLflow defaults to the local server at
`http://127.0.0.1:5000` and experiment `pydantic-ai-rlm-synthetic-benchmark`.
Override those values with `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME`, or
the corresponding command-line flags.

```powershell
.\.venv\Scripts\python.exe benchmarks\live_synthetic_suite.py `
  --output benchmark-results\live-suite.json
```

Open `http://127.0.0.1:5000` after the run and select the experiment named above.
All workload context is synthetic. MLflow autologging records prompts, generated
code, tool results, model responses, latency, and token usage; it does not receive
the OpenRouter API key. End-to-end timing and parent RSS include tracing overhead.

The provider-reported token and cost totals currently cover main-agent requests
only. Nested `llm_query` calls are counted, but their token usage and cost are not
aggregated by the current implementation. Input data is synthetic, and credential
values are checked for absence from captured tool output and final answers.

## Imported large evaluation cases

`live_large_cases.py` ports two deterministic workloads from the earlier local
`pydantic-ai-rlm-harness/test-project` into this repository's current engine:

| Case | Records | UTF-8 bytes | Integrity |
|---|---:|---:|---|
| Customer feedback | 5,000 | 1,557,639 | SHA-256 `aebfea7895c38ecace581a7feb69ee2b7e59677aa5df7db8645add9d1d75e915` |
| Accounting/audit | 3,682 | 1,654,972 | SHA-256 `92a537609a9fb97251a98040c00479dce09bab9453abf6037c061fc671756d58` |

The original business prompts, deterministic generators, and factual quality
gates are retained. Generated JSONL is digest-checked before use and remains
ignored under `benchmarks/large_cases/data/`. Every run is traced to the local
MLflow experiment `pydantic-ai-rlm-large-benchmark`, including the main agent,
provider requests, `execute_code`, and explicit nested `llm_query` spans.

The first measured baseline is recorded in
[`results/2026-09-15-large-cases.md`](results/2026-09-15-large-cases.md).
The passing post-remediation runs and lifecycle measurements are recorded in
[`results/2026-09-15-output-recovery.md`](results/2026-09-15-output-recovery.md).

Run both cases and retain the report:

```powershell
.\.venv\Scripts\python.exe benchmarks\live_large_cases.py `
  --output benchmark-results\live-large-cases.json
```

Use `--case customer_feedback` or `--case accounting` to run one case. The
quality gates validate factual conclusions, bounded tool/model trajectories,
focused submodel exposure for feedback, worker cleanup, complete MLflow traces,
and absence of the OpenRouter API key from outputs and serialized traces.

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
