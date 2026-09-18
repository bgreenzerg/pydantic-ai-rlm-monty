# Security model

This fork treats model-generated Python as hostile. It replaces the upstream
in-process CPython `exec`/`eval` REPL with Pydantic Monty 0.0.22 and creates a
fresh native worker process and interpreter session for every Pydantic AI run.
The original RLM prompts remain byte-for-byte unchanged.

## Trust boundaries

- The application, this package, its dependency environment and the selected
  Monty binary are trusted deployment components.
- The main model, generated Python, input context and sub-model output are
  untrusted data.
- Model providers are external data processors. The main provider receives user
  prompts and tool results. If `sub_model` is configured, generated code may send
  selected context to that provider through `llm_query()`.
- Host logs and telemetry are a separate data sink. Content logging is off by
  default.

Monty is an application sandbox, not a complete hostile-tenant boundary for a
compromised native runtime. A regulated deployment should still run the service
with an unprivileged OS identity, outbound network policy, read-only application
files and container or VM isolation appropriate to its threat model.

## Implemented controls

- No CPython `exec`, `eval`, dynamic import hook, global `chdir`, or replacement
  of process-wide stdout/stderr.
- No filesystem mount, host object, environment, socket, subprocess or OS
  callback is exposed to generated code.
- The only optional external callback is `llm_query`; it has call-count, prompt,
  response, provider timeout, token and suspension limits. Prompt size is checked
  inside Monty before the value can cross the worker/host boundary.
- Context accepts only bounded JSON-like trees or strings. Structured input is
  serialized under a streaming byte budget and reconstructed before use so
  mutable host objects are detached. Exact built-in types are required; hostile
  container subclasses are rejected without invoking their conversion hooks.
- Code, context, memory, recursion, cumulative execution time, checkout time and
  concurrent worker count are bounded. Printed output has a smaller soft return
  budget and a separate finite hard emission budget.
- Crossing the soft output budget discards excess bytes and reports truncation
  without destroying persistent REPL state. A hard output flood, memory limit,
  terminal execution limit or worker crash poisons and closes the session.
- Final expression values and exception messages are converted to bounded output
  inside Monty. They cannot bypass output controls through the worker protocol.
- The initial context transfer is a separate safe feed. A syntax error in the
  model's first snippet therefore cannot discard context before it is loaded.
- Fatal sandbox conditions propagate as `SandboxFatalError`, invalidating the
  complete agent run instead of allowing the model to return an unsupported
  normal answer. Any retry must start a new run and worker from immutable input.
- Final answers are checked for internally contradictory simple numeric
  equations by a restricted AST/`Decimal` parser. The validator never executes
  model text, has bounded linear scanning and AST limits, and requests a corrected
  model response when it finds a mismatch.
- A final answer requires a successful `execute_code` result by default; ordinary
  code failures do not satisfy this evidence gate. Grounded responses additionally
  require exact source quotes and consistent consecutive citation markers.
- Tool calls within a run are sequential. A new worker is created lazily on the
  first code call, and `__aexit__` performs deterministic cleanup even when a
  request is cancelled. Runs without code calls consume no worker capacity.
- The process-wide worker ceiling defaults to four. Set
  `PYDANTIC_AI_RLM_MAX_SESSIONS` before import to select 1–1024 slots.
  All threads share the ceiling; each application process has its own ceiling.
  Admission times out after `checkout_timeout`; pending request memory is not
  bounded by this ceiling. Size capacity using workload and memory measurements.
- Code, prompts and outputs are redacted from package logs unless
  `include_content=True` is deliberately selected.
- The executable is resolved from an explicit absolute path or the installed
  `pydantic-monty-runtime` package manifest, never from `PATH` or an environment
  override. Package-managed workers are automatically verified against the
  manifest's SHA-256; an optional deployment allow-list adds an independent pin.

## Review of the upstream implementation

The upstream 0.1.2 implementation had several properties that are unsuitable for
multi-tenant regulated workloads:

- Model code ran through CPython `exec`/`eval` and retained access to imports and
  host file APIs.
- Capturing output and changing directories modified process-global state.
- REPLs lived in a process-global dictionary keyed by task identity and depended
  on manual cleanup. Identifier reuse and missed cleanup created cross-run data
  retention and cross-tenant risk.
- `asyncio.wait_for` could time out the caller while the executor thread and
  generated code continued running.
- Session namespaces and temporary resources could outlive the request.
- Verbose logging emitted generated code, tool results and sub-model content.

The replacement has no global session registry and owns all tenant state inside
the run-scoped toolset and its one-use worker. Validated context is not copied a
second time when handed to the sandbox environment. The environment's reference
is cleared after the first successful feed and on every close path. The one
run-scoped dependency reference remains only until final-output validation (where
grounding quotes are checked), then becomes collectible with the run.

## Production deployment requirements

Before production approval:

1. Build from a reviewed commit and use a lock file with approved package hashes.
2. Store the Monty worker in a read-only, administrator-owned location; configure
   its absolute path and approved SHA-256 digest.
3. Run as an unprivileged service identity in a hardened container or VM. Deny
   outbound traffic by default and allow only approved model endpoints.
4. Keep `sub_model=None` unless its provider, region, retention policy and data
   classification are approved. Treat `llm_query` as deliberate data egress.
5. Keep content logging disabled. Apply redaction and access control to application,
   provider and infrastructure telemetry as well.
6. Set lower per-workload limits where possible and enforce service-level request,
   tenant, cost and rate limits outside this library.
7. Run SAST, dependency audit, SBOM/signing, container scanning, penetration tests,
   load/soak tests and incident-response exercises in the target environment.
8. Re-review every Pydantic AI or Monty update. The narrow dependency bounds are
   intentional; do not auto-upgrade the sandbox runtime without validation.

The arithmetic validator is defence in depth, not a general factual verifier.
Regulated workflows must additionally validate task-specific invariants and
source provenance before committing financial or operational decisions.

Prefer the async agent API for production requests. The sync compatibility API
passes a provider timeout, but a non-conforming provider implementation cannot be
forcibly stopped safely inside the caller's Python thread; enforce an outer
service-process deadline if sync sub-model calls are unavoidable.

## Validation snapshot

The release validation date, checks, live workloads, trace IDs and measured
resource use for version 0.2.1 are recorded in [RELEASE_NOTES.md](RELEASE_NOTES.md).
The lifecycle benchmark and live results are reproducible from the scripts linked
in [benchmarks/README.md](benchmarks/README.md).

These results support engineering review; they are not certification, a formal
penetration test, or an authorization to process central-bank data.

The Windows worker installed from the tested PyPI wheel is not Authenticode
signed and has no embedded vendor/version metadata. Its local SHA-256 was
`7bef8adbc8c1dc87a19b32f3a0e76430bc99d33a93e2a1d382a8421a68bea3d8`.
Treat that value only as evidence for this exact local artifact—not as a vendor
trust statement. An internal, reviewed artifact-signing and provenance process is
a production prerequisite for this native executable.
