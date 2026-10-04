from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .expression import Expression
from .operators import BINARY, UNARY_PARAMS, UNARY_SIMPLE

CANONICAL_OPS = ("lag", "diff", "mean", "std", "add", "sub")
CANONICAL_PARAMS = {
    "lag": (1,),
    "diff": (1,),
    "mean": (5,),
    "std": (5,),
}
ALL_SERIES_OPS = tuple(UNARY_SIMPLE) + tuple(UNARY_PARAMS) + tuple(BINARY) + ("where",)
OPS = ALL_SERIES_OPS + ("gt",)


@dataclass(frozen=True)
class Grammar:
    raw_variables: tuple[str, ...]
    max_local_depth: int = 2
    terminal_probability: float = 0.65
    operators: tuple[str, ...] = CANONICAL_OPS
    param_choices: Mapping[str, tuple[int, ...]] | None = None
    conditions: tuple[Expression, ...] = ()

    def __post_init__(self) -> None:
        if not self.raw_variables:
            raise ValueError("raw_variables cannot be empty")
        if self.max_local_depth < 1:
            raise ValueError("max_local_depth must be positive")
        if not 0.0 <= self.terminal_probability < 1.0:
            raise ValueError("terminal_probability must lie in [0, 1)")
        if any(op not in ALL_SERIES_OPS for op in self.operators):
            raise ValueError("unknown grammar operator")
        if "where" in self.operators and not self.conditions:
            raise ValueError("where requires at least one observable condition")

    def sample(
        self,
        rng: np.random.Generator,
        *,
        concept_ids: Sequence[str] = (),
        transition_weights: Mapping[tuple[str, str], float] | None = None,
    ) -> Expression:
        terminals = self._terminals(concept_ids)
        return self._series(
            rng,
            depth=self.max_local_depth,
            terminals=terminals,
            transition_weights=transition_weights,
            parent_op="__root__",
            force_op=True,
        )

    def _terminals(self, concept_ids: Sequence[str]) -> tuple[Expression, ...]:
        return (
            tuple(Expression.raw(name) for name in self.raw_variables)
            + tuple(Expression.concept(name) for name in concept_ids)
        )

    def _choose_op(
        self,
        rng: np.random.Generator,
        choices: tuple[str, ...],
        parent_op: str,
        transition_weights: Mapping[tuple[str, str], float] | None,
    ) -> str:
        if not transition_weights:
            return str(rng.choice(choices))
        weights = np.array(
            [
                1.0 + max(float(transition_weights.get((parent_op, op), 0.0)), 0.0)
                for op in choices
            ],
            dtype=np.float64,
        )
        return str(rng.choice(choices, p=weights / weights.sum()))

    def _params(self, op: str) -> tuple[int, ...]:
        if self.param_choices and op in self.param_choices:
            return self.param_choices[op]
        return UNARY_PARAMS[op]

    def _series(
        self,
        rng: np.random.Generator,
        *,
        depth: int,
        terminals: tuple[Expression, ...],
        transition_weights: Mapping[tuple[str, str], float] | None,
        parent_op: str,
        force_op: bool = False,
    ) -> Expression:
        if depth == 0 or (not force_op and rng.random() < self.terminal_probability):
            return terminals[int(rng.integers(len(terminals)))]

        choices = tuple(
            op
            for op in self.operators
            if op != "where" or depth >= 2
        )
        op = self._choose_op(rng, choices, parent_op, transition_weights)
        child_depth = depth - 1

        def child(parent: str) -> Expression:
            return self._series(
                rng,
                depth=child_depth,
                terminals=terminals,
                transition_weights=transition_weights,
                parent_op=parent,
            )

        if op in UNARY_SIMPLE:
            return Expression.unary(op, child(op))
        if op in UNARY_PARAMS:
            return Expression.unary(op, child(op), int(rng.choice(self._params(op))))
        if op in BINARY:
            return Expression.binary(op, child(op), child(op))

        condition = self.conditions[int(rng.integers(len(self.conditions)))]
        left = terminals[int(rng.integers(len(terminals)))]
        right = terminals[int(rng.integers(len(terminals)))]
        return Expression.where(condition, left, right)


def proposal_pool(
    raw_variables: Sequence[str],
    concept_ids: Sequence[str] = (),
) -> tuple[Expression, ...]:
    raws = tuple(Expression.raw(name) for name in raw_variables)
    terminals = raws + tuple(Expression.concept(name) for name in concept_ids)
    unique: dict[str, Expression] = {}

    for terminal in terminals:
        for op in UNARY_SIMPLE:
            expression = Expression.unary(op, terminal)
            unique[expression.key] = expression
        for op, params in UNARY_PARAMS.items():
            for param in params:
                expression = Expression.unary(op, terminal, param)
                unique[expression.key] = expression

    for left in terminals:
        for right in terminals:
            for op in BINARY:
                expression = Expression.binary(op, left, right)
                unique[expression.key] = expression

    conditions = tuple(
        Expression.compare(left, right)
        for left in raws
        for right in raws
        if left.key != right.key
    )
    for condition in conditions:
        for left in terminals:
            for right in terminals:
                expression = Expression.where(condition, left, right)
                unique[expression.key] = expression

    return tuple(unique[key] for key in sorted(unique))


def concept_pool(
    raw_variables: Sequence[str],
    concept_ids: Sequence[str],
) -> tuple[Expression, ...]:
    ids = set(concept_ids)
    return tuple(
        expression
        for expression in proposal_pool(raw_variables, concept_ids)
        if ids.intersection(expression.concept_ids())
    )


def operator_counts(expression: Expression) -> dict[str, int]:
    counts: dict[str, int] = {}
    if expression.op not in {"raw", "concept"}:
        counts[expression.op] = counts.get(expression.op, 0) + 1
    for arg in expression.args:
        for op, count in operator_counts(arg).items():
            counts[op] = counts.get(op, 0) + count
    return counts
