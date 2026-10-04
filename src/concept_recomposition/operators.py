from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .expression import Expression

FloatArray = NDArray[np.float64]
WINDOWS = (3, 5)
UNARY = ("diff", "lag", "mean", "std", "sign", "abs")
BINARY = ("add", "sub", "mul", "ratio", "min", "max")


def _lag(x: FloatArray, periods: int) -> FloatArray:
    out = np.full_like(x, np.nan)
    out[periods:] = x[:-periods]
    return out


def _diff(x: FloatArray, periods: int) -> FloatArray:
    out = np.full_like(x, np.nan)
    out[periods:] = x[periods:] - x[:-periods]
    return out


def _rolling(x: FloatArray, window: int, stat: str) -> FloatArray:
    out = np.full_like(x, np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(x, window)
    finite = np.isfinite(windows)
    count = finite.sum(axis=1)
    safe = np.where(finite, windows, 0.0)
    if stat == "mean":
        values = np.divide(
            safe.sum(axis=1),
            count,
            out=np.full(count.shape, np.nan, dtype=float),
            where=count > 0,
        )
    else:
        mean = np.divide(
            safe.sum(axis=1),
            count,
            out=np.zeros(count.shape, dtype=float),
            where=count > 0,
        )
        centered = np.where(finite, windows - mean[:, None], 0.0)
        values = np.sqrt(
            np.divide(
                np.square(centered).sum(axis=1),
                count - 1,
                out=np.full(count.shape, np.nan, dtype=float),
                where=count > 1,
            )
        )
    out[window - 1 :] = values
    return out


def evaluate(
    expr: Expression,
    variables: dict[str, FloatArray],
    concepts: dict[str, FloatArray],
    cache: dict[Expression, FloatArray] | None = None,
) -> FloatArray:
    if cache is not None and expr in cache:
        return cache[expr]

    if expr.kind == "var":
        result = variables[expr.name]
    elif expr.kind == "concept":
        result = concepts[expr.name]
    elif expr.kind == "unary":
        x = evaluate(expr.args[0], variables, concepts, cache)
        if expr.name == "diff":
            result = _diff(x, int(expr.param or 1))
        elif expr.name == "lag":
            result = _lag(x, int(expr.param or 1))
        elif expr.name in {"mean", "std"}:
            result = _rolling(x, int(expr.param or 3), expr.name)
        elif expr.name == "sign":
            result = np.sign(x)
        elif expr.name == "abs":
            result = np.abs(x)
        else:
            raise ValueError(expr.name)
    elif expr.kind == "where":
        cond = evaluate(expr.args[0], variables, concepts, cache)
        left = evaluate(expr.args[1], variables, concepts, cache)
        right = evaluate(expr.args[2], variables, concepts, cache)
        result = np.where(cond > 0, left, right)
    else:
        left = evaluate(expr.args[0], variables, concepts, cache)
        right = evaluate(expr.args[1], variables, concepts, cache)
        if expr.name == "add":
            result = left + right
        elif expr.name == "sub":
            result = left - right
        elif expr.name == "mul":
            result = left * right
        elif expr.name == "ratio":
            result = np.divide(
                left,
                right,
                out=np.full_like(left, np.nan),
                where=np.abs(right) > 1e-8,
            )
        elif expr.name == "min":
            result = np.minimum(left, right)
        elif expr.name == "max":
            result = np.maximum(left, right)
        else:
            raise ValueError(expr.name)

    if cache is not None:
        cache[expr] = result
    return result
