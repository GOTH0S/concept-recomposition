from __future__ import annotations

import numpy as np

from .expression import Expression
from .operators import evaluate
from .worlds import World


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 20:
        return 0.0
    x = left[mask] - left[mask].mean()
    y = right[mask] - right[mask].mean()
    scale = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if scale < 1e-12:
        return 0.0
    return float(abs(np.dot(x, y) / scale))


def score_expression(
    expression: Expression,
    world: World,
    concepts: dict[str, Expression],
    split: int | None = None,
    cache: dict[str, np.ndarray] | None = None,
) -> tuple[float, float]:
    values = np.asarray(evaluate(expression, world.data, concepts, cache), dtype=np.float64)
    boundary = world.validation_end if split is None else split
    validation = max(_corr(values[:boundary], target[:boundary]) for target in world.outcomes)
    heldout = max(_corr(values[boundary:], target[boundary:]) for target in world.outcomes)
    return validation, heldout
