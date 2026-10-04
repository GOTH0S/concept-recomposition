from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from .expression import Expression

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

UNARY_PARAMS = {
    "lag": (1, 5, 20),
    "diff": (1, 5, 20),
    "mean": (3, 5, 10, 20),
    "std": (5, 10, 20),
}
UNARY_SIMPLE = ("abs", "sign", "rank")
BINARY = ("add", "sub", "mul", "ratio", "min", "max")


def _shift(values: FloatArray, periods: int) -> FloatArray:
    out = np.full_like(values, np.nan)
    if periods < values.size:
        out[periods:] = values[:-periods]
    return out


def _rolling(values: FloatArray, window: int, std: bool) -> FloatArray:
    out = np.full_like(values, np.nan)
    finite = np.where(np.isfinite(values), values, 0.0)
    valid = np.isfinite(values).astype(np.int64)
    count = np.concatenate(([0], np.cumsum(valid)))
    total = np.concatenate(([0.0], np.cumsum(finite)))
    sumsq = np.concatenate(([0.0], np.cumsum(finite * finite)))
    n = count[window:] - count[:-window]
    sums = total[window:] - total[:-window]
    means = np.divide(sums, n, out=np.full_like(sums, np.nan), where=n > 0)
    if not std:
        out[window - 1 :] = np.where(n == window, means, np.nan)
        return out
    squares = sumsq[window:] - sumsq[:-window]
    variance = np.divide(
        squares - np.divide(sums * sums, n, out=np.zeros_like(sums), where=n > 0),
        n,
        out=np.full_like(sums, np.nan),
        where=n > 0,
    )
    out[window - 1 :] = np.where(n == window, np.sqrt(np.maximum(variance, 0.0)), np.nan)
    return out


def _causal_rank(values: FloatArray) -> FloatArray:
    out = np.full_like(values, np.nan)
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        history = values[: index + 1]
        history = history[np.isfinite(history)]
        below = np.count_nonzero(history < value)
        equal = np.count_nonzero(history == value)
        out[index] = 2.0 * ((below + 0.5 * equal) / history.size) - 1.0
    return out


def evaluate(
    expr: Expression,
    data: Mapping[str, FloatArray],
    concepts: Mapping[str, Expression] | None = None,
    cache: dict[str, FloatArray | BoolArray] | None = None,
) -> FloatArray | BoolArray:
    concepts = concepts or {}
    cache = cache if cache is not None else {}
    if expr.key in cache:
        return cache[expr.key]

    if expr.op == "raw":
        out = np.asarray(data[str(expr.value)], dtype=np.float64)
        if out.ndim != 1:
            raise ValueError(f"{expr.value} must be one-dimensional")
    elif expr.op == "concept":
        out = evaluate(concepts[str(expr.value)], data, concepts, cache)
    elif expr.op == "abs":
        out = np.abs(evaluate(expr.args[0], data, concepts, cache))
    elif expr.op == "sign":
        out = np.sign(evaluate(expr.args[0], data, concepts, cache))
    elif expr.op == "rank":
        out = _causal_rank(np.asarray(evaluate(expr.args[0], data, concepts, cache)))
    elif expr.op == "lag":
        out = _shift(np.asarray(evaluate(expr.args[0], data, concepts, cache)), int(expr.param))
    elif expr.op == "diff":
        values = np.asarray(evaluate(expr.args[0], data, concepts, cache))
        out = values - _shift(values, int(expr.param))
    elif expr.op == "mean":
        out = _rolling(
            np.asarray(evaluate(expr.args[0], data, concepts, cache)),
            int(expr.param),
            False,
        )
    elif expr.op == "std":
        out = _rolling(
            np.asarray(evaluate(expr.args[0], data, concepts, cache)),
            int(expr.param),
            True,
        )
    elif expr.op in BINARY:
        left = np.asarray(evaluate(expr.args[0], data, concepts, cache), dtype=np.float64)
        right = np.asarray(evaluate(expr.args[1], data, concepts, cache), dtype=np.float64)
        if left.shape != right.shape:
            raise ValueError("binary operands must have matching shapes")
        if expr.op == "add":
            out = left + right
        elif expr.op == "sub":
            out = left - right
        elif expr.op == "mul":
            out = left * right
        elif expr.op == "ratio":
            out = np.divide(
                left,
                right,
                out=np.full_like(left, np.nan),
                where=np.abs(right) > 1e-9,
            )
        elif expr.op == "min":
            out = np.minimum(left, right)
        else:
            out = np.maximum(left, right)
    elif expr.op == "gt":
        left = np.asarray(evaluate(expr.args[0], data, concepts, cache), dtype=np.float64)
        right = np.asarray(evaluate(expr.args[1], data, concepts, cache), dtype=np.float64)
        out = np.isfinite(left) & np.isfinite(right) & (left > right)
    elif expr.op == "where":
        condition = np.asarray(evaluate(expr.args[0], data, concepts, cache), dtype=np.bool_)
        left = np.asarray(evaluate(expr.args[1], data, concepts, cache), dtype=np.float64)
        right = np.asarray(evaluate(expr.args[2], data, concepts, cache), dtype=np.float64)
        out = np.where(condition, left, right)
    else:
        raise ValueError(f"unknown operator: {expr.op}")

    cache[expr.key] = out
    return out
