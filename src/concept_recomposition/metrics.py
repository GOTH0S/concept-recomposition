from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .search import SearchResult


def _available_concepts(result: SearchResult, budget: int) -> set[str]:
    available: set[str] = set()
    for row in result.archive.rows():
        generation = int(row["generation"])
        if result.generation_ends[generation] <= budget:
            available.add(str(row["concept_id"]))
    return available


def _prefix_metrics(result: SearchResult, budget: int) -> dict[str, float]:
    records = [record for record in result.records if record.proposal <= budget]
    exact = [record for record in records if record.exact_target]
    first = min((record.proposal for record in exact), default=None)
    distinct = {record.target_key for record in exact if record.target_key is not None}
    available = _available_concepts(result, budget)

    target_refs: set[str] = set()
    for record in exact:
        target_refs.update(record.concept_refs)
    useful = result.archive.lineage(tuple(target_refs)) & available if target_refs else set()

    false_expansion = 0.0
    if available:
        false_expansion = 1.0 - len(useful) / len(available)

    return {
        "reached_target": float(bool(exact)),
        "first_target": float(first) if first is not None else float("nan"),
        "distinct_targets": float(len(distinct)),
        "concepts": float(len(available)),
        "false_expansion": false_expansion,
        "best_validation": max((record.validation_score for record in records), default=0.0),
        "best_heldout": max((record.heldout_score for record in records), default=0.0),
    }


def summarize(
    results: Iterable[SearchResult],
    budget: int,
) -> dict[str, float]:
    values = [_prefix_metrics(result, budget) for result in results]
    if not values:
        raise ValueError("results cannot be empty")

    first = np.array([value["first_target"] for value in values], dtype=float)
    return {
        "reach_rate": float(np.mean([value["reached_target"] for value in values])),
        "mean_first_target": (
            float(np.nanmean(first)) if np.isfinite(first).any() else float("nan")
        ),
        "mean_distinct_targets": float(
            np.mean([value["distinct_targets"] for value in values])
        ),
        "mean_concepts": float(np.mean([value["concepts"] for value in values])),
        "mean_false_expansion": float(
            np.mean([value["false_expansion"] for value in values])
        ),
        "mean_best_validation": float(
            np.mean([value["best_validation"] for value in values])
        ),
        "mean_best_heldout": float(
            np.mean([value["best_heldout"] for value in values])
        ),
    }
