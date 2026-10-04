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
    expr: Expression, data: dict[str, FloatArray], rng: np.random.Generator
) -> FloatArray:
    signal = evaluate(expr, data)
    scale = np.nanstd(signal)
    return signal + rng.normal(scale=max(scale, 1e-6) * 0.08, size=signal.size)


def build_world(name: str, seed: int = 0, n: int = 360) -> World:
    data = _raw_data(seed, n)
    rng = np.random.default_rng(seed + 10_000)
    x1, x2, x3, _, x5 = (Expression.raw(f"x{i}") for i in range(1, 6))

    if name == "shallow":
        targets = (Expression.unary("diff", x1, 5),)
    elif name == "deep":
        z = Expression.unary("mean", Expression.unary("abs", x2), 5)
        targets = (Expression.binary("add", z, Expression.unary("diff", z, 5)),)
    elif name == "reuse":
        z = Expression.unary("mean", Expression.unary("abs", x3), 5)
        targets = (
            Expression.binary("add", z, Expression.unary("diff", z, 5)),
            Expression.binary(
                "sub", Expression.unary("mean", z, 3), Expression.unary("lag", z, 1)
            ),
        )
    elif name == "context":
        z = Expression.unary("mean", Expression.unary("abs", x2), 5)
        targets = (Expression.where(x5, z, Expression.unary("diff", z, 5)),)
    elif name == "decoy":
        targets = ()
    else:
        raise ValueError(f"unknown world: {name}")

    if targets:
        outcomes = tuple(_with_noise(target, data, rng) for target in targets)
    else:
        outcomes = (rng.normal(size=n),)
    return World(name, data, targets, outcomes)


WORLD_NAMES = ("shallow", "deep", "reuse", "context", "decoy")
