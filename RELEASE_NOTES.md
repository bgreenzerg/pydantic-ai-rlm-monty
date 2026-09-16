# Release notes

## 0.2.1 — release hardening

Version 0.2.1 closes the nine P1 findings from the package release review. The
upstream RLM prompt text remains byte-for-byte unchanged.

### Security, correctness and resource controls

1. Final expression values, exception messages and `llm_query` arguments are
   bounded inside the Monty worker before they cross the worker/host protocol.
2. Context is loaded through a separate safe feed, so a syntax error in the first
   generated snippet no longer consumes or loses the run context.
3. Structured context validation traverses lazily, serializes under the configured
   byte budget, rejects non-built-in container/scalar subclasses and avoids a
   second detached copy when the run-scoped toolset creates its environment.
4. Arithmetic validation now uses bounded linear discovery, a node-limited
   iterative AST evaluator and source-exact `Decimal` literals. Pathological
   input cannot trigger recursive evaluation or quadratic regex behavior.
5. Monty workers are acquired lazily on the first code call. Runs without tool
   execution no longer occupy one of the four process-wide worker slots.
6. Final answers require at least one successful `execute_code` result by default.
   Grounded responses also verify marker/key agreement, consecutive numbering and
   verbatim source membership for every quote.
7. `RLMDependencies.config.code_timeout` is preserved unless the caller supplies
   an explicit factory override; the previous implicit 60-second overwrite is
   removed.
8. The base package now installs Pydantic AI's OpenAI provider support required by
   the default `openai:gpt-5` model string.
9. Package version and build outputs are advanced to 0.2.1, preventing stale
   0.2.0 wheel/source-distribution ambiguity.

### Compatibility notes

- `RLMConfig.require_code_execution` defaults to `True`. Set it to `False` only
  when an agent is intentionally allowed to answer without sandbox evidence.
- `create_rlm_agent(..., code_timeout=None)` and `create_rlm_toolset(...,
  code_timeout=None)` now mean “use the run's `RLMConfig.code_timeout`”. Explicit
  numeric overrides continue to work.
- Structured context accepts exact built-in `str`, `dict`, `list`, numeric,
  boolean and null values. Subclasses are rejected at the trust boundary.
- Grounding quotes must be 10–200 characters and occur verbatim in context.

### Verification

The release gate covers unit, adversarial, concurrency and end-to-end agent
tests; Ruff; mypy; Bandit; dependency audit; clean wheel/sdist installation; and
both large synthetic OpenRouter workloads with local MLflow trace read-back.
Exact commands and final results are filled in from the release branch before the
pull request is merged.

The local release-to-release performance gate passed 40 iterations with 1 MiB
context per run. Active sandbox p95 improved from 48.991 ms to 48.112 ms, while
no-tool p50 improved from 47.526 ms to 7.306 ms and peak Monty workers fell from
one to zero. See
[`benchmarks/results/2026-09-16-release-hardening.md`](benchmarks/results/2026-09-16-release-hardening.md).
