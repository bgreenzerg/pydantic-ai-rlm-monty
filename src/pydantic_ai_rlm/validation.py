from __future__ import annotations

import ast
import re
from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import Any

from .dependencies import ContextType

_CLAIMED = re.compile(r"\s*(?P<number>[+-]?(?:\d{1,30}(?:\.\d{1,18})?|\.\d{1,18}))(?P<percent>\s*%?)")
_MARKER = re.compile(r"\[(\d+)]")
_ARITHMETIC_CHARS = frozenset("0123456789.+-*/() \t\r\n")
_MAX_EXPRESSION_CHARS = 512
_MAX_EXPRESSION_NODES = 256
_MAX_EQUATIONS = 100


def _decimal_literal(expression: str, node: ast.Constant) -> Decimal:
    literal = ast.get_source_segment(expression, node)
    if literal is None:
        raise ValueError("missing numeric literal")
    return Decimal(literal)


def _decimal_expression(expression: str) -> Decimal:  # noqa: C901
    """Evaluate a small numeric AST iteratively and without Python ``eval``."""
    root = ast.parse(expression, mode="eval")
    stack: list[tuple[ast.AST, bool]] = [(root, False)]
    values: dict[int, Decimal] = {}
    nodes = 0
    while stack:
        node, visited = stack.pop()
        if not visited:
            nodes += 1
            if nodes > _MAX_EXPRESSION_NODES:
                raise ValueError("arithmetic expression is too complex")
            stack.append((node, True))
            if isinstance(node, ast.Expression):
                stack.append((node.body, False))
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                stack.append((node.operand, False))
            elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                stack.extend(((node.right, False), (node.left, False)))
            elif not (isinstance(node, ast.Constant) and type(node.value) in (int, float)):
                raise ValueError("unsupported arithmetic expression")
            continue

        if isinstance(node, ast.Expression):
            values[id(node)] = values[id(node.body)]
        elif isinstance(node, ast.Constant):
            values[id(node)] = _decimal_literal(expression, node)
        elif isinstance(node, ast.UnaryOp):
            operand = values[id(node.operand)]
            values[id(node)] = operand if isinstance(node.op, ast.UAdd) else -operand
        elif isinstance(node, ast.BinOp):
            left = values[id(node.left)]
            right = values[id(node.right)]
            if isinstance(node.op, ast.Add):
                values[id(node)] = left + right
            elif isinstance(node.op, ast.Sub):
                values[id(node)] = left - right
            elif isinstance(node.op, ast.Mult):
                values[id(node)] = left * right
            else:
                values[id(node)] = left / right
    return values[id(root)]


def _left_expression(text: str, equals_at: int) -> str | None:
    start = equals_at
    lower_bound = max(0, equals_at - _MAX_EXPRESSION_CHARS)
    while start > lower_bound and text[start - 1] in _ARITHMETIC_CHARS:
        start -= 1
    expression = text[start:equals_at].strip()
    if not expression or not any(operator in expression for operator in "+-*/"):
        return None
    return expression


def arithmetic_consistency_errors(text: str, *, max_errors: int = 5) -> list[str]:
    """Find contradictory bounded arithmetic equations without evaluating model text as code."""
    if max_errors <= 0:
        return []
    normalized = (
        text[:1_000_000]
        .replace("\N{MINUS SIGN}", "-")
        .replace("\N{EN DASH}", "-")
        .replace("\N{MULTIPLICATION SIGN}", "*")
        .replace("\\times", "*")
        .replace("\\%", "%")
    )
    errors: list[str] = []
    equations = 0
    for equals in re.finditer("=", normalized):
        equations += 1
        if equations > _MAX_EQUATIONS:
            break
        expression = _left_expression(normalized, equals.start())
        if expression is None:
            continue
        claimed_match = _CLAIMED.match(normalized, equals.end())
        if claimed_match is None:
            continue
        claimed_text = claimed_match.group("number")
        try:
            calculated = _decimal_expression(expression)
            claimed = Decimal(claimed_text)
        except (SyntaxError, ValueError, DivisionByZero, InvalidOperation, RecursionError, MemoryError):
            continue
        decimal_places = len(claimed_text.partition(".")[2])
        tolerance = Decimal("0.5") * (Decimal(10) ** -decimal_places)
        candidates = (calculated, calculated * 100) if claimed_match.group("percent").strip() else (calculated,)
        closest = min(candidates, key=lambda candidate: abs(candidate - claimed))
        if abs(closest - claimed) > tolerance:
            errors.append(f"{expression}={claimed_text} (calculated {closest.normalize()})")
            if len(errors) >= max_errors:
                break
    return errors


def grounding_consistency_errors(  # noqa: C901
    info: str,
    grounding: dict[str, str],
    context: ContextType,
    *,
    max_errors: int = 5,
) -> list[str]:
    """Validate citation numbering and exact quote membership in a JSON-like context."""
    if max_errors <= 0:
        return []
    errors: list[str] = []
    referenced = set(_MARKER.findall(info))
    keys = set(grounding)
    if not referenced:
        errors.append("grounded response contains no citation markers")
    if referenced != keys:
        errors.append("citation markers and grounding keys do not match")
    if keys:
        expected = {str(index) for index in range(1, len(keys) + 1)}
        if keys != expected:
            errors.append("citation keys must be consecutive starting at 1")

    valid_quotes: dict[str, str] = {}
    for key, quote in grounding.items():
        if not isinstance(quote, str) or not 10 <= len(quote) <= 200:
            errors.append(f"citation {key} must be an exact quote of 10-200 characters")
        else:
            valid_quotes[key] = quote
        if len(errors) >= max_errors:
            return errors[:max_errors]

    missing = _quotes_missing_from_context(set(valid_quotes.values()), context)
    for key, quote in valid_quotes.items():
        if quote in missing:
            errors.append(f"citation {key} is not an exact quote from context")
            if len(errors) >= max_errors:
                break
    return errors[:max_errors]


def _quotes_missing_from_context(quotes: set[str], context: ContextType) -> set[str]:
    if not quotes:
        return set()
    missing = set(quotes)
    stack: list[Any] = [context]
    while stack and missing:
        value = stack.pop()
        if type(value) is str:
            missing = {quote for quote in missing if quote not in value}
        elif type(value) is list:
            stack.extend(value)
        elif type(value) is dict:
            stack.extend(value.keys())
            stack.extend(value.values())
        elif value is not None:
            rendered = str(value)
            missing = {quote for quote in missing if quote not in rendered}
    return missing
