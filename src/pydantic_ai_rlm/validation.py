from __future__ import annotations

import ast
import re
from decimal import Decimal, DivisionByZero, InvalidOperation

_NUMBER = r"(?:\d{1,30}(?:\.\d{1,18})?|\.\d{1,18})"
_SIGNED_ATOM = rf"[+-]?(?:{_NUMBER}|\(\s*[+-]?{_NUMBER}\s*\))"
_EQUATION = re.compile(
    rf"(?<![\w.])(?P<expression>{_SIGNED_ATOM}(?:\s*[+*/-]\s*{_SIGNED_ATOM})+)"
    rf"\s*=\s*(?P<claimed>[+-]?{_NUMBER})(?P<percent>\s*%?)"
)


def _decimal_expression(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return _decimal_expression(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _decimal_expression(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _decimal_expression(node.left)
        right = _decimal_expression(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError("unsupported arithmetic expression")


def arithmetic_consistency_errors(text: str, *, max_errors: int = 5) -> list[str]:
    """Find contradictory simple arithmetic equations without evaluating model text as code."""
    normalized = (
        text.replace("\N{MINUS SIGN}", "-")
        .replace("\N{EN DASH}", "-")
        .replace("\N{MULTIPLICATION SIGN}", "*")
        .replace("\\times", "*")
        .replace("\\%", "%")
    )
    errors: list[str] = []
    for match in _EQUATION.finditer(normalized[:1_000_000]):
        expression = match.group("expression")
        claimed_text = match.group("claimed")
        try:
            parsed = ast.parse(expression, mode="eval")
            calculated = _decimal_expression(parsed)
            if match.group("percent").strip():
                calculated *= 100
            claimed = Decimal(claimed_text)
        except (SyntaxError, ValueError, DivisionByZero, InvalidOperation):
            continue
        decimal_places = len(claimed_text.partition(".")[2])
        tolerance = Decimal("0.5") * (Decimal(10) ** -decimal_places)
        if abs(calculated - claimed) > tolerance:
            errors.append(f"{expression}={claimed_text} (calculated {calculated.normalize()})")
            if len(errors) >= max_errors:
                break
    return errors
