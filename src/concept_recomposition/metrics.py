from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .search import SearchResult


def summarize(results: Iterable[SearchResult]) -> dict[str, float]:
    values = list(results)
    if not values:
        raise ValueError("results cannot be empty")

    reached = np.array([result.reached_target for result in values], dtype=float)
    first = np.array(
        [result.first_target_proposal or np.nan for result in values],
        dtype=float,
    )
    return {
        "reach_rate": float(reached.mean()),
        "mean_first_target": (
            float(np.nanmean(first)) if np.isfinite(first).any() else float("nan")
        ),
        "mean_distinct_targets": float(
            np.mean([len(result.distinct_targets) for result in values])
        ),
        "mean_concepts": float(np.mean([len(result.archive) for result in values])),
        "mean_false_expansion": float(
            np.mean([result.archive.false_expansion_rate() for result in values])
        ),
        "mean_best_validation": float(
            np.mean([result.best_validation for result in values])
        ),
        "mean_best_heldout": float(np.mean([result.best_heldout for result in values])),
    }
