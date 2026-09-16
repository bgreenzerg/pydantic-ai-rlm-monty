# Large-case benchmark baseline — 2026-09-15

This report records the first run of the two deterministic synthetic workloads
ported from `C:\Python\pydantic-ai-rlm-harness\test-project`. The production
implementation under `src/` was not changed for this run.

## Reproduction

```powershell
.\.venv\Scripts\python.exe benchmarks\live_large_cases.py `
  --output benchmark-results\live-large-cases.json
```

- Model: `openrouter:openai/gpt-5.6-luna`
- MLflow: `http://127.0.0.1:5000`
- Experiment: `pydantic-ai-rlm-large-benchmark` (ID `2`)
- Suite ID: `f2ec2843-f471-484c-97e3-0ba0d5d95c5c`
- Raw local report: `benchmark-results/live-large-cases.json` (intentionally
  ignored because each rerun replaces it)

Both generated JSONL fixtures matched the byte sizes and SHA-256 digests of the
original harness fixtures.

## Results

| Case | Quality | Time | Main requests | Input/output tokens | Main cost | Peak parent RSS | Peak Monty RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Customer feedback (5,000 records; 1,557,639 bytes) | FAIL | 56.048 s | 11 | 48,906 / 3,834 | $0.00750306 | 193.982 MB | 22.913 MB |
| Accounting (3,682 records; 1,654,972 bytes) | FAIL | 41.615 s | 10 | 34,588 / 2,162 | $0.00413474 | 200.892 MB | 16.552 MB |
| **Suite** | **0/2** | **102.889 s** | **21** | **83,494 / 5,996** | **$0.01163780** | — | — |

The parent process started at 129.884 MB RSS and ended at 199.750 MB, a 69.865
MB increase including MLflow client state. No Monty worker remained after either
case. Three MLflow/Git helper processes remained and are reported separately
from sandbox workers.

Nested `llm_query` usage was zero in both cases, so reported provider tokens and
cost are complete for this run. Credentials were absent from final answers,
tool outputs, and serialized traces.

## Failure analysis

The feedback trace is `tr-5ac7cfe0f0a85b6d8cc7fe1afbb94e9b`. The agent found
the correct broad segment, but did not perform the required independent
`llm_query` review and estimated 55–65 direct reports instead of the planted,
exact count of 24. It also described an "independent focused review" despite
making zero nested model calls.

The accounting trace is `tr-4d2f76b1c17a501dc9f0501c54601fd0`. The agent printed
one line for every parsed record while exploring. The tool exceeded the
configured 10,000-byte collector bound (`10052 bytes > 10000 bytes`), which
raised `MemoryError` and permanently closed the stateful Monty session. Eight of
nine tool calls therefore failed and none of the five questions was answered.

The feedback run reached the same condition later (`10079 bytes > 10000 bytes`)
after printing a 120-record candidate set. Five of ten tool calls failed.

This differs materially from the older harness configuration: its 10,000
character result limit truncated oversized tool output after execution. In the
current implementation, `max_output_bytes` is a fail-closed collector limit and
an overflow poisons the whole REPL session. The current agent instructions also
say to use `print()` liberally and do not disclose the hard terminal output
limit. This combination is the immediate cause of both failed trajectories.

The fail-closed behavior is safe for containment, but the controller does not
fail the overall agent run when the sandbox becomes unavailable. It may instead
return unsupported conclusions or a graceful-looking partial answer. That is a
production-readiness issue even though worker cleanup and credential isolation
passed in these runs.
