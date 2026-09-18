import os
import subprocess
import sys

import pytest

from pydantic_ai_rlm import RLMConfig, RLMDependencies
from pydantic_ai_rlm.repl import _sandbox_capacity_from_env


@pytest.mark.parametrize("value", ["", "0", "-1", "32.0", "1025", "True", " 32", "\uff13\uff12", "9" * 10000])
def test_invalid_capacity(monkeypatch, value):
    monkeypatch.setenv("PYDANTIC_AI_RLM_MAX_SESSIONS", value)
    with pytest.raises(ValueError, match="PYDANTIC_AI_RLM_MAX_SESSIONS"):
        _sandbox_capacity_from_env()


def test_default_capacity(monkeypatch):
    monkeypatch.delenv("PYDANTIC_AI_RLM_MAX_SESSIONS", raising=False)
    assert _sandbox_capacity_from_env() == 4


@pytest.mark.parametrize("capacity", [1, 32, 1024])
def test_fresh_process_capacity(capacity):
    code = """
import os
from pydantic_ai_rlm.repl import _SANDBOX_GATE
capacity = int(os.environ['PYDANTIC_AI_RLM_MAX_SESSIONS'])
assert all(_SANDBOX_GATE.acquire(blocking=False) for _ in range(capacity))
assert not _SANDBOX_GATE.acquire(blocking=False)
os.environ['PYDANTIC_AI_RLM_MAX_SESSIONS'] = '2'
for _ in range(capacity):
    _SANDBOX_GATE.release()
assert all(_SANDBOX_GATE.acquire(blocking=False) for _ in range(capacity))
assert not _SANDBOX_GATE.acquire(blocking=False)
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, timeout=30, env={**os.environ, "PYDANTIC_AI_RLM_MAX_SESSIONS": str(capacity)}
    )


def test_multikey_context():
    context = {"a": "one", "b": "two", "nested": [{"c": 3}, {"d": 4}]}
    assert RLMDependencies(context=context).context == context


def test_sibling_depth_does_not_accumulate():
    context = [[1], [2], [3], [4]]
    assert RLMDependencies(context=context, config=RLMConfig(max_context_depth=2)).context == context
    with pytest.raises(ValueError, match="max_context_depth"):
        RLMDependencies(context=[[1], [[2]]], config=RLMConfig(max_context_depth=2))


def test_sibling_error_path():
    with pytest.raises(TypeError, match=r"context.b\[0\].bad"):
        RLMDependencies(context={"a": [1], "b": [{"bad": object()}]})
