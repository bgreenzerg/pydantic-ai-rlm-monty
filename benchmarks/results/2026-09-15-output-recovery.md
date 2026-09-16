# Large-case output recovery verification — 2026-09-15

This report verifies the runtime remediation for the terminal output-overflow
behavior recorded in `2026-09-15-large-cases.md`. The deterministic source
fixtures and original business prompts remain unchanged.

## Runtime verification

With a 10,000-byte soft return limit and a 4 MiB hard emission limit:

| Reproduced operation | Emitted | Returned | Next persistent-state check |
|---|---:|---:|---:|
| Print metadata for all 3,682 accounting records | 587,093 bytes | 10,000 bytes | `len(recs) == 3682` |
| Print all 120 narrowed feedback candidates | 18,726 bytes | 10,000 bytes | `len(focus) == 120` |

Both oversized outputs completed successfully, included an explicit truncation
notice, and retained usable REPL state. A separate hard-limit regression proves
that an actual output flood invalidates the complete agent run and cannot be
followed by a normal model answer.

## Live OpenRouter and MLflow verification

| Case | Result | Time | Key trajectory evidence | MLflow trace |
|---|---:|---:|---|---|
| Customer feedback, 5,000 records | PASS | 81.030 s | 10 code calls, 1 soft truncation, 2 completed `llm_query` calls, 2.648% evidence exposure | `tr-9930d3d7ba989bc7fa80b34c43f86ca4` |
| Accounting, 3,682 records | PASS | 71.568 s | 10 code calls, correct answers to Q1–Q5, no terminal span | `tr-c9f2afd3412efec6654f4eb72ab9a3c4` |

Both traces were read back from the local MLflow experiment
`pydantic-ai-rlm-large-benchmark`. Their factual gates pass, their top-level
trace status is `OK`, no terminal agent/provider spans are present, the API key
is absent, and no Monty worker remains after completion.

A separate accounting run exposed the internally inconsistent equation
`14.6 + 3.5 = 17.1`. The new deterministic final-answer validator detects that
specific failure without `eval`, using a restricted arithmetic AST and
`Decimal`. An integration test proves that Pydantic suppresses such an answer,
asks the model to recompute it, and only returns the corrected result.

## Lifecycle and soak verification

- 100 fresh sandboxes, each receiving 1,654,972 bytes of context:
  - startup/execution/cleanup p50: 29.431 ms
  - p95: 30.688 ms
  - maximum: 32.034 ms
  - parent RSS delta after collection: 90,112 bytes
  - leaked child processes: 0
- 50 consecutive 100,001-byte prints in one stateful sandbox:
  - all 50 were soft-truncated successfully
  - persistent counter finished at 50
  - parent RSS delta after collection: 1,323,008 bytes
  - remaining Monty workers: 0

These measurements are local engineering evidence, not a formal production
certification or penetration test.
