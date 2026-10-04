from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .expression import Expression
from .operators import BINARY, UNARY_PARAMS, UNARY_SIMPLE

SERIES_OPS = tuple(UNARY_SIMPLE) + tuple(UNARY_PARAMS) + tuple(BINARY) + ("where",)
OPS = SERIES_OPS + ("gt",)


@dataclass(frozen=True)
class Grammar:
    raw_variables: tuple[str, ...]
    max_local_depth: int = 2
    terminal_probability: float = 0.35

    def __post_init__(self) -> None:
        if not self.raw_variables:
            raise ValueError("raw_variables cannot be empty")
        if self.max_local_depth < 1:
            raise ValueError("max_local_depth must be positive")
        if not 0.0 <= self.terminal_probability < 1.0:
            raise ValueError("terminal_probability must lie in [0, 1)")

    def sample(
        self,
        rng: np.random.Generator,
        *,
        concept_ids: Sequence[str] = (),
        op_weights: Mapping[str, float] | None = None,
    ) -> Expression:
        terminals = self._terminals(concept_ids)
        return self._series(
            rng,
            depth=self.max_local_depth,
            terminals=terminals,
            op_weights=op_weights,
            force_op=True,
        )

    def _terminals(self, concept_ids: Sequence[str]) -> tuple[Expression, ...]:
        raws = tuple(Expression.raw(name) for name in self.raw_variables)
        concepts = tuple(Expression.concept(name) for name in concept_ids)
        return raws + concepts

    def _choose_op(
        self,
        rng: np.random.Generator,
        op_weights: Mapping[str, float] | None,
    ) -> str:
        if not op_weights:
            return str(rng.choice(SERIES_OPS))
        weights = np.array(
            [max(float(op_weights.get(op, 1.0)), 1e-6) for op in SERIES_OPS],
            dtype=np.float64,
        )
        return str(rng.choice(SERIES_OPS, p=weights / weights.sum()))

    def _series(
        self,
        rng: np.random.Generator,
        *,
        depth: int,
        terminals: tuple[Expression, ...],
        op_weights: Mapping[str, float] | None,
        force_op: bool = False,
    ) -> Expression:
        if depth == 0 or (not force_op and rng.random() < self.terminal_probability):
            return terminals[int(rng.integers(len(terminals)))]

        op = self._choose_op(rng, op_weights)
        child_depth = depth - 1

        if op in UNARY_SIMPLE:
            return Expression.unary(
                op,
                self._series(
                    rng,
                    depth=child_depth,
                    terminals=terminals,
                    op_weights=op_weights,
                ),
            )

        if op in UNARY_PARAMS:
            param = int(rng.choice(UNARY_PARAMS[op]))
            return Expression.unary(
                op,
                self._series(
                    rng,
                    depth=child_depth,
                    terminals=terminals,
                    op_weights=op_weights,
                ),
                param,
            )

        if op in BINARY:
            return Expression.binary(
                op,
                self._series(
                    rng,
                    depth=child_depth,
                    terminals=terminals,
                    op_weights=op_weights,
                ),
                self._series(
                    rng,
                    depth=child_depth,
                    terminals=terminals,
                    op_weights=op_weights,
                ),
            )

        condition = Expression.compare(
            self._series(
                rng,
                depth=child_depth,
                terminals=terminals,
                op_weights=op_weights,
            ),
            self._series(
                rng,
                depth=child_depth,
                terminals=terminals,
                op_weights=op_weights,
            ),
        )
        return Expression.where(
            condition,
            self._series(
                rng,
                depth=child_depth,
                terminals=terminals,
                op_weights=op_weights,
            ),
            self._series(
                rng,
                depth=child_depth,
                terminals=terminals,
                op_weights=op_weights,
            ),
        )


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


def sample_pool(
    rng: np.random.Generator,
    pool: Sequence[Expression],
    count: int,
    op_weights: Mapping[str, float] | None = None,
) -> tuple[Expression, ...]:
    if count <= 0 or not pool:
        return ()
    if count >= len(pool):
        return tuple(pool)

    if op_weights:
        weights = np.array(
            [max(float(op_weights.get(expression.op, 1.0)), 1e-6) for expression in pool],
            dtype=np.float64,
        )
        probabilities = weights / weights.sum()
    else:
        probabilities = None

    indices = rng.choice(len(pool), size=count, replace=False, p=probabilities)
    return tuple(pool[int(index)] for index in indices)


def operator_counts(expression: Expression) -> dict[str, int]:
    counts: dict[str, int] = {}
    if expression.op not in {"raw", "concept"}:
        counts[expression.op] = counts.get(expression.op, 0) + 1
    for arg in expression.args:
        for op, count in operator_counts(arg).items():
            counts[op] = counts.get(op, 0) + count
    return counts
