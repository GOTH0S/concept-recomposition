from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .expression import Expression
from .operators import evaluate

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class World:
    name: str
    data: dict[str, FloatArray]
    targets: tuple[Expression, ...]
    outcomes: tuple[FloatArray, ...]
    intermediates: tuple[Expression, ...] = ()
    validation_end: int = 360

    @property
    def target_keys(self) -> frozenset[str]:
        return frozenset(target.key for target in self.targets)


def _ar1(rng: np.random.Generator, n: int, phi: float) -> FloatArray:
    values = np.empty(n, dtype=np.float64)
    values[0] = rng.normal()
    for index in range(1, n):
        values[index] = phi * values[index - 1] + rng.normal(scale=0.8)
    return values


def _raw_data(seed: int, n: int) -> dict[str, FloatArray]:
    rng = np.random.default_rng(seed)
    return {f"x{i}": _ar1(rng, n, 0.75 - i * 0.05) for i in range(1, 6)}


def _with_noise(
    expression: Expression,
    data: dict[str, FloatArray],
    rng: np.random.Generator,
) -> FloatArray:
    signal = np.asarray(evaluate(expression, data), dtype=np.float64)
    scale = np.nanstd(signal)
    return signal + rng.normal(scale=max(scale, 1e-6) * 0.10, size=signal.size)


def build_world(name: str, seed: int = 0, n: int = 600) -> World:
    data = _raw_data(seed, n)
    rng = np.random.default_rng(seed + 10_000)
    x1, x2, x3, _, x5 = (Expression.raw(f"x{i}") for i in range(1, 6))
    intermediates: tuple[Expression, ...] = ()

    if name == "shallow":
        targets = (Expression.unary("diff", x1, 5),)
    elif name == "deep":
        z = Expression.unary("mean", Expression.unary("diff", x2, 5), 10)
        intermediates = (z,)
        targets = (Expression.unary("mean", z, 3),)
    elif name == "reuse":
        z = Expression.unary("mean", Expression.unary("diff", x3, 1), 5)
        intermediates = (z,)
        targets = (
            Expression.unary("mean", z, 3),
            Expression.unary("lag", z, 1),
        )
    elif name == "context":
        z = Expression.unary("mean", Expression.unary("diff", x2, 1), 5)
        intermediates = (z,)
        condition = Expression.compare(x5, x1)
        targets = (Expression.where(condition, z, x2),)
    elif name == "decoy":
        targets = ()
    else:
        raise ValueError(f"unknown world: {name}")

    outcomes = (
        tuple(_with_noise(target, data, rng) for target in targets)
        if targets
        else (rng.normal(size=n),)
    )
    return World(
        name=name,
        data=data,
        targets=targets,
        outcomes=outcomes,
        intermediates=intermediates,
    )


WORLD_NAMES = ("shallow", "deep", "reuse", "context", "decoy")
