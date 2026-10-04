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
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(abs(np.corrcoef(x, y)[0, 1]))


def score_expression(
    expression: Expression,
    world: World,
    concepts: dict[str, Expression],
    split: int = 240,
) -> tuple[float, float]:
    values = evaluate(expression, world.data, concepts)
    validation = max(_corr(values[:split], target[:split]) for target in world.outcomes)
    heldout = max(_corr(values[split:], target[split:]) for target in world.outcomes)
    return validation, heldout
