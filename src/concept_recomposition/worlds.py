from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .expression import Expression
from .operators import evaluate

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class World:
    name: str
    variables: dict[str, FloatArray]
    target: Expression | None
    target_signal: FloatArray | None
    y: FloatArray
    description: str


def _base(seed: int, n: int) -> dict[str, FloatArray]:
    rng = np.random.default_rng(seed)
    return {
        f"x{i}": rng.normal(size=n).cumsum() / 8.0 + rng.normal(size=n)
        for i in range(1, 6)
    }


def make_world(
    name: str,
    *,
    seed: int,
    n: int = 240,
    noise: float = 0.20,
) -> World:
    variables = _base(seed, n)
    rng = np.random.default_rng(seed + 10_000)

    if name == "contextual":
        variables["x5"] = variables["x5"] + 1.0

    if name == "shallow":
        target = Expression.unary("diff", Expression.var("x1"), 3)
        description = "A one-step transformation is enough."
    elif name == "deep":
        z = Expression.unary("mean", Expression.var("x2"), 5)
        target = Expression.unary("mean", z, 3)
        description = "The target needs a useful intermediate transformation."
    elif name == "branching":
        z = Expression.unary("mean", Expression.var("x3"), 5)
        target = Expression.binary("add", z, z)
        description = "One intermediate structure is reused twice."
    elif name == "contextual":
        z = Expression.unary("mean", Expression.var("x4"), 5)
        target = Expression.where(
            Expression.var("x5"),
            z,
            Expression.var("x4"),
        )
        description = "A derived object matters only in one observed state."
    elif name == "decoy":
        target = None
        description = "There is no reusable hidden structure."
    else:
        raise ValueError(name)

    target_signal = None if target is None else evaluate(target, variables, {})
    if target_signal is None:
        y = rng.normal(size=n)
    else:
        scale = np.nanstd(target_signal)
        y = target_signal + rng.normal(
            scale=noise * (scale if scale > 0 else 1.0),
            size=n,
        )
    return World(name, variables, target, target_signal, y, description)


WORLD_NAMES = ("shallow", "deep", "branching", "contextual", "decoy")
