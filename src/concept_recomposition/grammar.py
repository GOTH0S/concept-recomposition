from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .expression import Expression
from .operators import BINARY, UNARY_PARAMS, UNARY_SIMPLE

OPS = tuple(UNARY_SIMPLE) + tuple(UNARY_PARAMS) + tuple(BINARY) + ("where",)


def proposal_pool(
    raw_variables: Sequence[str],
    concept_ids: Sequence[str] = (),
) -> tuple[Expression, ...]:
    raws = tuple(Expression.raw(name) for name in raw_variables)
    terminals = raws + tuple(Expression.concept(name) for name in concept_ids)
    unique: dict[str, Expression] = {}

    for terminal in terminals:
        for op in UNARY_SIMPLE:
            expr = Expression.unary(op, terminal)
            unique[expr.key] = expr
        for op, params in UNARY_PARAMS.items():
            for param in params:
                expr = Expression.unary(op, terminal, param)
                unique[expr.key] = expr

    for left in terminals:
        for right in terminals:
            for op in BINARY:
                expr = Expression.binary(op, left, right)
                unique[expr.key] = expr

    for condition in raws:
        for left in terminals:
            for right in terminals:
                expr = Expression.where(condition, left, right)
                unique[expr.key] = expr

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
            [max(float(op_weights.get(expr.op, 1.0)), 1e-6) for expr in pool],
            dtype=np.float64,
        )
        probabilities = weights / weights.sum()
    else:
        probabilities = None

    indices = rng.choice(
        len(pool),
        size=count,
        replace=False,
        p=probabilities,
    )
    return tuple(pool[int(index)] for index in indices)


def operator_counts(expr: Expression) -> dict[str, int]:
    counts: dict[str, int] = {}
    if expr.op not in {"raw", "concept"}:
        counts[expr.op] = counts.get(expr.op, 0) + 1
    for arg in expr.args:
        for op, count in operator_counts(arg).items():
            counts[op] = counts.get(op, 0) + count
    return counts
