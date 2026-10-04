from __future__ import annotations

from collections import defaultdict

from .expression import Expression


def operator_transitions(expression: Expression) -> tuple[tuple[str, str], ...]:
    transitions: list[tuple[str, str]] = []

    def visit(node: Expression, parent: str) -> None:
        if node.op in {"raw", "concept"}:
            return
        transitions.append((parent, node.op))
        for child in node.args:
            visit(child, node.op)

    visit(expression, "__root__")
    return tuple(transitions)


class MotifModel:
    def __init__(self) -> None:
        self._weights: dict[tuple[str, str], float] = defaultdict(float)

    @property
    def weights(self) -> dict[tuple[str, str], float]:
        return dict(self._weights)

    def observe(self, expression: Expression, score: float) -> None:
        increment = max(score, 0.05)
        for transition in operator_transitions(expression):
            self._weights[transition] += increment
