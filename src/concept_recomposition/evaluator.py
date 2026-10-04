from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .expression import Expression
from .operators import evaluate
from .worlds import World


@dataclass(frozen=True)
class EvidenceScore:
    discovery: float
    confirmation: float
    heldout: float

    @property
    def promotion(self) -> float:
        return min(self.discovery, self.confirmation)


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


def _best_corr(values: np.ndarray, outcomes: tuple[np.ndarray, ...]) -> float:
    return max(_corr(values, target) for target in outcomes)


def score_expression(
    expression: Expression,
    world: World,
    concepts: dict[str, Expression],
    *,
    discovery_end: int,
    heldout_start: int,
    cache: dict[str, np.ndarray] | None = None,
) -> EvidenceScore:
    if not 20 <= discovery_end < heldout_start <= len(next(iter(world.data.values()))):
        raise ValueError("invalid evidence split")

    values = evaluate(expression, world.data, concepts, cache)
    discovery = _best_corr(
        values[:discovery_end],
        tuple(target[:discovery_end] for target in world.outcomes),
    )
    confirmation = _best_corr(
        values[discovery_end:heldout_start],
        tuple(target[discovery_end:heldout_start] for target in world.outcomes),
    )
    heldout = _best_corr(
        values[heldout_start:],
        tuple(target[heldout_start:] for target in world.outcomes),
    )
    return EvidenceScore(discovery, confirmation, heldout)
