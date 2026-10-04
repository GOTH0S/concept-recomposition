from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .expression import Expression
from .operators import BINARY, UNARY, WINDOWS


@dataclass(slots=True)
class ProposalPolicy:
    operator_weight: dict[str, float] = field(default_factory=dict)

    def update(self, expressions: list[Expression]) -> None:
        for expression in expressions:
            current = self.operator_weight.get(expression.name, 1.0)
            self.operator_weight[expression.name] = current + 0.5

    def weight(self, name: str) -> float:
        return self.operator_weight.get(name, 1.0)


def local_pool(
    terminals: tuple[Expression, ...],
    raw_variables: tuple[Expression, ...],
) -> tuple[Expression, ...]:
    candidates: set[Expression] = set()

    for terminal in terminals:
        for name in UNARY:
            if name in {"diff", "lag", "mean", "std"}:
                for window in WINDOWS:
                    candidates.add(Expression.unary(name, terminal, window))
            else:
                candidates.add(Expression.unary(name, terminal))

    for left in terminals:
        for right in terminals:
            for name in BINARY:
                candidates.add(Expression.binary(name, left, right))

    for condition in raw_variables:
        for when_positive in terminals:
            for otherwise in terminals:
                candidates.add(
                    Expression.where(
                        condition,
                        when_positive,
                        otherwise,
                    )
                )

    return tuple(sorted(candidates, key=str))


def sample_pool(
    rng: np.random.Generator,
    pool: tuple[Expression, ...],
    policy: ProposalPolicy,
    count: int,
) -> tuple[Expression, ...]:
    if count >= len(pool):
        return pool
    weights = np.array(
        [
            policy.weight(expression.name)
            * (
                1.0
                + 3.0
                * sum(arg.kind == "concept" for arg in expression.args)
            )
            for expression in pool
        ],
        dtype=float,
    )
    probabilities = weights / weights.sum()
    indices = rng.choice(
        len(pool),
        size=count,
        replace=False,
        p=probabilities,
    )
    return tuple(pool[int(index)] for index in indices)
