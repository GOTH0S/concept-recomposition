from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .expression import Expression
from .operators import BINARY, UNARY_PARAMS, UNARY_SIMPLE

OPS = tuple(UNARY_SIMPLE) + tuple(UNARY_PARAMS) + tuple(BINARY) + ("gt", "where")


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
