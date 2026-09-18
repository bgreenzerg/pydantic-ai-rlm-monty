<h1 align="center">Pydantic AI RLM Monty</h1>

<p align="center">
  <b>Handle Extremely Large Contexts with Any LLM Provider</b>
</p>

<p align="center">
  <a href="https://github.com/vstorm-co/pydantic-ai-rlm">Upstream repository</a> •
  <a href="https://pypi.org/project/pydantic-ai-rlm/">Upstream PyPI package</a> •
  <a href="https://github.com/vstorm-co/pydantic-ai-rlm#examples">Examples</a>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/pydantic/pydantic-ai"><img src="https://img.shields.io/badge/Powered%20by-Pydantic%20AI-E92063?logo=pydantic&logoColor=white" alt="Pydantic AI"></a>
  <a href="https://github.com/bgreenzerg/pydantic-ai-rlm-monty/releases"><img src="https://img.shields.io/github/v/release/bgreenzerg/pydantic-ai-rlm-monty" alt="GitHub release"></a>
</p>

<p align="center">
  <b>Switch Providers Instantly</b>
  &nbsp;•&nbsp;
  <b>Sandboxed Code Execution</b>
  &nbsp;•&nbsp;
  <b>Sub-Model Delegation</b>
  &nbsp;•&nbsp;
  <b>Grounded Citations</b>
  &nbsp;•&nbsp;
  <b>Fully Type-Safe</b>
</p>

---

> **Local Monty fork:** This branch replaces the original in-process CPython
> executor with a dedicated Pydantic Monty worker for every agent run. The
> upstream RLM instructions are byte-for-byte unchanged. See [SECURITY.md](SECURITY.md)
> before operating on regulated or confidential data.

## Concurrent Monty sessions

Set `PYDANTIC_AI_RLM_MAX_SESSIONS` before starting Python to configure the maximum
active sandbox sessions **per Python process** (default `4`, range `1`–`1024`):

```powershell
$env:PYDANTIC_AI_RLM_MAX_SESSIONS = "32"
python your_service.py
```

Linux/macOS: `PYDANTIC_AI_RLM_MAX_SESSIONS=32 python your_service.py`.
The value is read once on package import; restart processes to change it.
Invalid values raise `ValueError`. This is not a per-request `RLMConfig` option:
all threads and async runs share the limit. Each application process has its own.

Independent `agent.run()` calls can run concurrently using separate
`RLMDependencies` per request. Code calls within each run remain sequential.
A slot is acquired on the first code call and held until the run ends, including
model waits. Admission waits up to `RLMConfig(checkout_timeout=10)` seconds by
default (maximum 300), then raises `TimeoutError`. Waiting requests are not
bounded or guaranteed FIFO and are not automatically retried. Use a bounded
application queue to handle bursts.

Capacity is not a memory reservation: each sandbox has its own memory allowance
(256 MiB by default), with host context storage and worker overhead additional.
Size capacity against actual workloads and the number of application processes.
Run `python benchmarks/compare_parallel_capacity.py` for the four-versus-32
comparison with real Monty workers and simulated model wait time.

## What is RLM?

**RLM (Recursive Language Model)** is a pattern for handling contexts that exceed a model's context window, introduced by **Alex L. Zhang, Tim Kraska, and Omar Khattab** in their paper [Recursive Language Models](https://arxiv.org/abs/2512.24601). Instead of trying to fit everything into one prompt, the LLM writes Python code to programmatically explore and analyze the data.

**The key insight:** An LLM can write code to search through millions of lines in seconds, then use `llm_query()` to delegate semantic analysis of relevant chunks to a sub-model.

This library is an implementation inspired by the [original minimal implementation](https://github.com/alexzhang13/rlm-minimal).

---

## Get Started in 60 Seconds

```bash
python -m pip install https://github.com/bgreenzerg/pydantic-ai-rlm-monty/releases/download/v0.2.3/pydantic_ai_rlm_monty-0.2.3-py3-none-any.whl
```

```python
from pydantic_ai_rlm import run_rlm_analysis

answer = await run_rlm_analysis(
    context=massive_document,  # bounded by RLMConfig.max_context_bytes
    query="Find the magic number hidden in the text",
    model="openai:gpt-5",
    sub_model="openai:gpt-5-mini",
)
```

**That's it.** Your agent can now:

- Write Python code to analyze massive contexts
- Use `llm_query()` to delegate semantic analysis to sub-models
- Work with any Pydantic AI compatible provider

---

## Why pydantic-ai-rlm-monty?

### Switch Providers Instantly

Built on Pydantic AI, you can test any model with a single line change:

```python
# OpenAI
agent = create_rlm_agent(model="openai:gpt-5", sub_model="openai:gpt-5-mini")

# Anthropic
agent = create_rlm_agent(model="anthropic:claude-sonnet-4-5", sub_model="anthropic:claude-haiku-4-5")

agent = create_rlm_agent(model="anthropic:claude-sonnet-4-5", sub_model="openai:gpt-5-mini")
```

### Reusable Toolset

The RLM toolset integrates with any pydantic-ai agent:

```python
from pydantic_ai import Agent
from pydantic_ai_rlm import create_rlm_toolset, RLMDependencies

# Use the toolset in any agent
toolset = create_rlm_toolset(sub_model="openai:gpt-5-mini")
agent = Agent("openai:gpt-5", toolsets=[toolset])
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     pydantic-ai-rlm-monty                      │
│                                                                 │
│   ┌─────────────┐         ┌─────────────────────────────────┐   │
│   │   Main LLM  │         │     Sandboxed REPL Environment  │   │
│   │   (gpt-5)   │────────>│                                 │   │
│   └─────────────┘         │   context = <your massive data> │   │
│         │                 │                                 │   │
│         │                 │   # LLM writes Python code:     │   │
│         │                 │   for line in context.split():  │   │
│         │                 │       if "magic" in line:       │   │
│         │                 │           result = llm_query(   │   │
│         │                 │               f"Analyze: {line}"│   │
│         │                 │           )                     │   │
│         │                 │                                 │   │
│         │                 └───────────────┬─────────────────┘   │
│         │                                 │                     │
│         │                                 ▼                     │
│         │                       ┌─────────────────┐             │
│         │                       │    Sub LLM      │             │
│         │                       │  (gpt-5-mini)   │             │
│         │                       └─────────────────┘             │
│         │                                                       │
│         ▼                                                       │
│   ┌─────────────┐                                               │
│   │   Answer    │                                               │
│   └─────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

1. **Main LLM** receives the query and writes Python code
2. **REPL Environment** executes code with access to `context` variable
3. **llm_query()** delegates semantic analysis to a cheaper/faster sub-model
4. **Main LLM** synthesizes the final answer from code execution results

---

## Examples

### Live OpenRouter smoke test

With `OPENROUTER_API_KEY` and `ASSISTANT_MODEL=provider/model` in the ignored
local `.env` file, run a real main-model, Monty, and nested sub-model flow:

```powershell
python -m pip install -e ".[openrouter,dev]"
python scripts\live_openrouter_smoke.py
```

The workload contains synthetic data only. Logging reports sizes, timings and
status while leaving prompts, generated code and outputs redacted.

### Needle in Haystack

Find specific information in massive text:

```python
from pydantic_ai_rlm import RLMConfig, run_rlm_analysis

# 1 million lines of text with a hidden number
massive_text = generate_haystack(num_lines=1_000_000)

answer = await run_rlm_analysis(
    context=massive_text,
    query="Find the magic number hidden in the text",
    model="openai:gpt-5",
    sub_model="openai:gpt-5-mini",
    config=RLMConfig(
        max_context_bytes=128 * 1024 * 1024,
        max_memory_bytes=1024 * 1024 * 1024,
    ),
)
```

### JSON Data Analysis

Works with structured data too:

```python
from pydantic_ai_rlm import create_rlm_agent, RLMDependencies

agent = create_rlm_agent(model="openai:gpt-5")

deps = RLMDependencies(
    context={"users": [...], "transactions": [...], "logs": [...]},
)

result = await agent.run(
    "Find all users with suspicious transaction patterns",
    deps=deps,
)
```

### Grounded Responses with Citations

Get answers with traceable citations back to the source:

```python
from pydantic_ai_rlm import run_rlm_analysis

# Enable grounding for citation tracking
result = await run_rlm_analysis(
    context=financial_report,
    query="What were the key revenue changes?",
    model="openai:gpt-5",
    grounded=True,  # Returns GroundedResponse instead of str
)

# Response contains citation markers
print(result.info)
# "Revenue increased [1] primarily due to [2]"

# Grounding maps markers to exact quotes from the source
print(result.grounding)
# {"1": "by 45% year-over-year", "2": "expansion into Asian markets"}
```

Grounded output is validated before it is returned: markers and keys must match,
keys must be consecutive from `1`, and every 10–200 character quote must occur
verbatim in the supplied context. Invalid grounding triggers an output retry.

---

## API Reference

### `create_rlm_agent()`

Create a Pydantic AI agent with RLM capabilities.

```python
agent = create_rlm_agent(
    model="openai:gpt-5",           # Main model for orchestration
    sub_model="openai:gpt-5-mini",  # Model for llm_query() (optional)
    code_timeout=None,               # None uses RLMDependencies.config.code_timeout
    custom_instructions="...",       # Additional instructions
    grounded=True,                   # Return GroundedResponse with citations
)
```

### `create_rlm_toolset()`

Create a standalone RLM toolset for composition.

```python
toolset = create_rlm_toolset(
    code_timeout=60.0,
    sub_model="openai:gpt-5-mini",
)
```

### `run_rlm_analysis()` / `run_rlm_analysis_sync()`

Convenience functions for quick analysis.

```python
# Async
answer = await run_rlm_analysis(context, query, model="openai:gpt-5")

# Sync
answer = run_rlm_analysis_sync(context, query, model="openai:gpt-5")

# With grounding (returns GroundedResponse)
result = await run_rlm_analysis(context, query, grounded=True)
print(result.info)       # Text with [N] markers
print(result.grounding)  # {"1": "exact quote", ...}
```

### `RLMDependencies`

Dependencies for RLM agents.

```python
deps = RLMDependencies(
    context="...",  # str, dict, or list
    config=RLMConfig(
        code_timeout=60.0,
        truncate_output_chars=50_000,
        max_output_bytes=50_000,             # Soft output retained for the model
        max_emitted_output_bytes=4 * 1024 * 1024,  # Hard per-snippet flood limit
        validate_arithmetic=True,            # Retry inconsistent simple equations
        require_code_execution=True,         # Require successful sandbox evidence
        sub_model="openai:gpt-5-mini",
    ),
)
```

### `configure_logging()`

Enable verbose logging to see what the agent is doing in real-time.

```python
from pydantic_ai_rlm import configure_logging, run_rlm_analysis

# Enable logging (uses rich if installed, falls back to plain text)
configure_logging(enabled=True)  # metadata only; content stays redacted

# Explicitly enabling this can disclose prompts, code and model output to logs:
# configure_logging(enabled=True, include_content=True)

# Now you'll see code executions and outputs in the terminal
answer = await run_rlm_analysis(
    context=massive_document,
    query="Find the magic number",
    model="openai:gpt-5",
)

# Disable logging when done
configure_logging(enabled=False)
```

Install with rich logging support for syntax highlighting and styled output:

```bash
python -m pip install ".[logging]"
```

Or install rich separately:

```bash
pip install rich
```

With the secure default (`include_content=False`), you'll see sizes, status and
timings without content. With content logging explicitly enabled, you'll see:

- Syntax-highlighted code being executed (with rich)
- Execution results with status indicators (SUCCESS/ERROR)
- Execution time for each code block
- Variables created during execution
- LLM sub-queries and responses (when using `llm_query()`)

**Note:** Logging works without rich installed - it will use plain text output instead of styled panels

---

## REPL Environment

The sandboxed REPL provides:

| Feature | Description |
|---------|-------------|
| `context` variable | Your data loaded and ready to use |
| `llm_query(prompt)` | Delegate to sub-model (if configured) |
| Safe built-ins | `print`, `len`, `range`, etc. |
| Supported imports | Monty's capability-free Python subset |
| Persistent state | Variables persist across executions |
| Output capture | stdout/stderr is soft-truncated without discarding REPL state |

`max_output_bytes` is the soft response budget: output beyond it is discarded
and the agent receives an explicit truncation notice while sandbox variables
remain usable. `max_emitted_output_bytes` is a separate hard flood limit. A
snippet exceeding that limit, the memory limit, or a terminal execution limit
invalidates the complete agent run; callers receive `SandboxFatalError` instead
of an apparently successful answer. Retry such a request only as a new run with
a fresh sandbox and the original immutable input.

By default, a final answer is accepted only after at least one successful
`execute_code` result in that run. A failed snippet does not count. Set
`require_code_execution=False` only for an intentional non-RLM use case.

Final string answers also receive a bounded deterministic consistency check for
simple arithmetic equations. It parses only numeric arithmetic into a restricted
AST and evaluates with `Decimal`; model text is never executed. An inconsistent
equation triggers a Pydantic output retry. This catches transcription mistakes,
but it is not a substitute for domain-specific validation of source selection,
formulas, or conclusions.

A worker is started lazily on the first `execute_code` call and destroyed at run
completion. Runs that never execute code consume no worker slot. The sandbox has
no filesystem mounts, environment variables, sockets, subprocesses, or ambient
host objects. `llm_query()` is the only optional egress path and is absent unless
a sub-model is explicitly configured.

Resource limits are finite by default. Production deployments should pin the
Monty executable with `RLMConfig(monty_binary_path=..., monty_binary_sha256=...)`
and lock all Python dependencies. Full controls and residual risks are documented
in [SECURITY.md](SECURITY.md).

---

## Related Projects

- **[rlm](https://github.com/alexzhang13/rlm)** - Original RLM implementation by Alex L. Zhang, Tim Kraska, and Omar Khattab
- **[rlm-minimal](https://github.com/alexzhang13/rlm-minimal)** - Minal RLM implementation by Alex L. Zhang
- **[pydantic-ai](https://github.com/pydantic/pydantic-ai)** - The foundation: Agent framework by Pydantic
- **[pydantic-deep](https://github.com/vstorm-co/pydantic-deepagents)** - Full deep agent framework with planning, filesystem, and more

---

## Contributing

```bash
git switch codex/monty-sandbox
python -m pip install -e ".[dev]"
pytest
```

---

## License

MIT — see [LICENSE](LICENSE)

<p align="center">
  <sub>Built with Pydantic AI by <a href="https://github.com/vstorm-co">vstorm-co</a></sub>
</p>
