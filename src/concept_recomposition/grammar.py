from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .expression import Expression
from .operators import BINARY, UNARY_PARAMS, UNARY_SIMPLE

OPS = tuple(UNARY_SIMPLE) + tuple(UNARY_PARAMS) + tuple(BINARY) + ("where",)


def _weighted_choice(
    rng: np.random.Generator,
    items: Sequence[str],
    weights: Mapping[str, float] | None,
) -> str:
    if not weights:
        return str(rng.choice(items))
    raw = np.array([max(weights.get(item, 1.0), 1e-6) for item in items], dtype=float)
    return str(rng.choice(items, p=raw / raw.sum()))


def random_expression(
    rng: np.random.Generator,
    raw_variables: Sequence[str],
    concept_ids: Sequence[str],
    max_size: int,
    op_weights: Mapping[str, float] | None = None,
) -> Expression:
    terminals = [Expression.raw(name) for name in raw_variables]
    terminals.extend(Expression.concept(name) for name in concept_ids)

    def build(budget: int) -> Expression:
        if budget <= 1 or rng.random() < 0.28:
            return terminals[int(rng.integers(len(terminals)))]

        viable = list(UNARY_SIMPLE) + list(UNARY_PARAMS)
        if budget >= 3:
            viable.extend(BINARY)
        if budget >= 4:
            viable.append("where")
        op = _weighted_choice(rng, viable, op_weights)

        if op in UNARY_SIMPLE:
            return Expression.unary(op, build(budget - 1))
        if op in UNARY_PARAMS:
            param = int(rng.choice(UNARY_PARAMS[op]))
            return Expression.unary(op, build(budget - 1), param)
        if op in BINARY:
            left_budget = int(rng.integers(1, budget - 1))
            right_budget = budget - 1 - left_budget
            if right_budget < 1:
                right_budget = 1
                left_budget = budget - 2
            return Expression.binary(op, build(left_budget), build(right_budget))

        remaining = budget - 1
        cuts = sorted(rng.integers(1, remaining, size=2))
        budgets = (
            max(1, cuts[0]),
            max(1, cuts[1] - cuts[0]),
            max(1, remaining - cuts[1]),
        )
        return Expression.where(*(build(part) for part in budgets))

    return build(max_size)


def operator_counts(expr: Expression) -> dict[str, int]:
    counts: dict[str, int] = {}
    if expr.op not in {"raw", "concept"}:
        counts[expr.op] = counts.get(expr.op, 0) + 1
    for arg in expr.args:
        for op, count in operator_counts(arg).items():
            counts[op] = counts.get(op, 0) + count
    return counts


def base_templates(raw_variables: Sequence[str]) -> list[Expression]:
    raws = [Expression.raw(name) for name in raw_variables]
    proposals: list[Expression] = []
    for raw in raws:
        absolute = Expression.unary("abs", raw)
        for window in UNARY_PARAMS["mean"]:
            proposals.append(Expression.unary("mean", absolute, window))
    for raw in raws:
        for op in UNARY_SIMPLE:
            proposals.append(Expression.unary(op, raw))
        for op, params in UNARY_PARAMS.items():
            for param in params:
                proposals.append(Expression.unary(op, raw, param))
    return proposals


def concept_proposals(
    rng: np.random.Generator,
    raw_variables: Sequence[str],
    concept_ids: Sequence[str],
    count: int,
    op_weights: Mapping[str, float] | None = None,
) -> list[Expression]:
    if not concept_ids or count <= 0:
        return []
    condition = Expression.raw("x5") if "x5" in raw_variables else Expression.raw(raw_variables[0])
    roots = ("add", "sub", "where")
    proposals: list[Expression] = []
    for _ in range(count):
        concept = Expression.concept(str(rng.choice(concept_ids)))
        root = _weighted_choice(rng, roots, op_weights)
        if root == "add":
            period = int(rng.choice((1, 5, 20)))
            proposals.append(
                Expression.binary("add", concept, Expression.unary("diff", concept, period))
            )
        elif root == "sub":
            window = int(rng.choice((3, 5, 10)))
            period = int(rng.choice((1, 5)))
            proposals.append(
                Expression.binary(
                    "sub",
                    Expression.unary("mean", concept, window),
                    Expression.unary("lag", concept, period),
                )
            )
        else:
            period = int(rng.choice((1, 5, 20)))
            proposals.append(
                Expression.where(condition, concept, Expression.unary("diff", concept, period))
            )
    return proposals
