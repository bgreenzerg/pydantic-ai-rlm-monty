from __future__ import annotations

import pytest

from pydantic_ai_rlm import RLMConfig, RLMDependencies


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("code_timeout", 0),
        ("code_timeout", float("inf")),
        ("code_timeout", 301),
        ("max_total_execution_seconds", 3601),
        ("checkout_timeout", 301),
        ("max_memory_bytes", -1),
        ("max_memory_bytes", 1.5),
        ("max_suspensions", True),
        ("checkout_timeout", float("nan")),
    ],
)
def test_limits_must_be_finite_and_positive(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        RLMConfig(**{field: value})  # type: ignore[arg-type]


def test_context_must_be_json_compatible() -> None:
    with pytest.raises(TypeError, match="unsupported value type"):
        RLMDependencies(context={"unsafe": object()})  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="keys must be strings"):
        RLMDependencies(context={1: "value"})  # type: ignore[dict-item]


def test_context_size_is_measured_as_utf8_bytes() -> None:
    config = RLMConfig(max_context_bytes=4)
    with pytest.raises(ValueError, match="max_context_bytes"):
        RLMDependencies(context="ååå", config=config)


def test_caller_config_is_not_mutated() -> None:
    config = RLMConfig(sub_model=None)
    RLMDependencies(context="x", config=config)
    assert config.sub_model is None


def test_context_is_detached_from_mutable_host_input() -> None:
    source = {"records": ["tenant-a"]}
    deps = RLMDependencies(context=source)
    source["records"][0] = "tenant-b"

    assert deps.context == {"records": ["tenant-a"]}


def test_cyclic_context_is_rejected_without_recursion() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cyclic or shared"):
        RLMDependencies(context=cyclic)


def test_context_depth_and_item_count_are_bounded() -> None:
    with pytest.raises(ValueError, match="max_context_depth"):
        RLMDependencies(context={"outer": {"inner": 1}}, config=RLMConfig(max_context_depth=1))

    with pytest.raises(ValueError, match="max_context_items"):
        RLMDependencies(context=[1, 2, 3], config=RLMConfig(max_context_items=3))


def test_binary_digest_must_be_canonical_sha256() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        RLMConfig(monty_binary_sha256="A" * 64)


def test_submodel_calls_cannot_exceed_suspension_budget() -> None:
    with pytest.raises(ValueError, match="max_submodel_calls"):
        RLMConfig(max_submodel_calls=2, max_suspensions=1)


def test_hard_output_limit_cannot_be_smaller_than_soft_limit() -> None:
    with pytest.raises(ValueError, match="max_emitted_output_bytes"):
        RLMConfig(max_output_bytes=2048, max_emitted_output_bytes=1024)


def test_arithmetic_validation_flag_must_be_boolean() -> None:
    with pytest.raises(TypeError, match="validate_arithmetic"):
        RLMConfig(validate_arithmetic="yes")  # type: ignore[arg-type]


def test_code_execution_requirement_must_be_boolean() -> None:
    with pytest.raises(TypeError, match="require_code_execution"):
        RLMConfig(require_code_execution="yes")  # type: ignore[arg-type]


def test_context_rejects_container_and_scalar_subclasses() -> None:
    class HostList(list[object]):
        pass

    class HostString(str):
        pass

    with pytest.raises(TypeError, match="context must be"):
        RLMDependencies(context=HostList())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="unsupported value type"):
        RLMDependencies(context=[HostString("secret")])


def test_encoded_context_stops_at_configured_budget() -> None:
    with pytest.raises(ValueError, match="max_context_bytes"):
        RLMDependencies(context=["x" * 10_000], config=RLMConfig(max_context_bytes=100))
