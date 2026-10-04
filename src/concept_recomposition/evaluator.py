from __future__ import annotations

import numpy as np

from .expression import Expression
from .operators import evaluate
from .worlds import World


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 20:
        return 0.0
    x = left[mask]
    y = right[mask]
    x = x - x.mean()
    y = y - y.mean()
    scale = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if scale < 1e-12:
        return 0.0
    return float(abs(np.dot(x, y) / scale))


def score_expression(
    expression: Expression,
    world: World,
    concepts: dict[str, Expression],
    split: int = 240,
    cache: dict[str, np.ndarray] | None = None,
) -> tuple[float, float]:
    values = evaluate(expression, world.data, concepts, cache)
    validation = max(_corr(values[:split], target[:split]) for target in world.outcomes)
    heldout = max(_corr(values[split:], target[split:]) for target in world.outcomes)
    return validation, heldout
