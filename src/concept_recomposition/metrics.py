from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def correlation(left: FloatArray, right: FloatArray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 20:
        return 0.0
    x = left[mask]
    y = right[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def score(
    signal: FloatArray,
    y: FloatArray,
    split: int,
) -> tuple[float, float]:
    return (
        correlation(signal[:split], y[:split]),
        correlation(signal[split:], y[split:]),
    )
