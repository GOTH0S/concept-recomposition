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
    intermediate = [record for record in records if record.exact_intermediate]
    available = _available_concepts(result, budget)

    target_refs: set[str] = set()
    for record in exact:
        target_refs.update(record.concept_refs)
    useful = result.archive.lineage(tuple(target_refs)) & available if target_refs else set()

    best_validation_record = max(
        records,
        key=lambda record: record.validation_score,
    )
    best_heldout = max(record.heldout_score for record in records)

    concept_rows = [
        row
        for row in result.archive.rows()
        if str(row["concept_id"]) in available
    ]
    promoted_intermediate = any(
        str(row["expanded_key"]) in result.promoted_intermediates
        for row in concept_rows
    )

    return {
        "reached_target": float(bool(exact)),
        "reached_intermediate": float(bool(intermediate)),
        "promoted_intermediate": float(promoted_intermediate),
        "first_intermediate": (
            float(min(record.proposal for record in intermediate))
            if intermediate
            else float("nan")
        ),
        "first_target": float(
            min(record.proposal for record in exact)
        ) if exact else float("nan"),
        "distinct_targets": float(
            len({record.target_key for record in exact})
        ),
        "concepts": float(len(available)),
        "false_expansion": (
            0.0 if not available else 1.0 - len(useful) / len(available)
        ),
        "reused_concepts": float(
            sum(int(row["proposal_uses"]) > 0 for row in concept_rows)
        ),
        "useful_reuses": float(
            sum(int(row["useful_uses"]) for row in concept_rows)
        ),
        "max_target_depth": float(
            max((record.expanded_depth for record in exact), default=0)
        ),
        "mean_positive_gain": float(
            np.mean(
                [
                    record.validation_gain
                    for record in records
                    if record.concept_refs and record.validation_gain > 0
                ]
            )
        ) if any(
            record.concept_refs and record.validation_gain > 0
            for record in records
        ) else 0.0,
        "best_validation": best_validation_record.validation_score,
        "selected_heldout": best_validation_record.heldout_score,
        "best_heldout": best_heldout,
        "selection_regret": best_heldout - best_validation_record.heldout_score,
    }


def summarize(
    results: Iterable[SearchResult],
    budget: int,
) -> dict[str, float]:
    values = [_prefix_metrics(result, budget) for result in results]
    if not values:
        raise ValueError("results cannot be empty")

    first = np.array([value["first_target"] for value in values], dtype=float)
    first_intermediate = np.array(
        [value["first_intermediate"] for value in values],
        dtype=float,
    )

    def mean(field: str) -> float:
        return float(np.mean([value[field] for value in values]))

    return {
        "reach_rate": mean("reached_target"),
        "intermediate_reach_rate": mean("reached_intermediate"),
        "intermediate_promotion_rate": mean("promoted_intermediate"),
        "mean_first_intermediate": (
            float(np.nanmean(first_intermediate))
            if np.isfinite(first_intermediate).any()
            else float("nan")
        ),
        "mean_first_target": (
            float(np.nanmean(first))
            if np.isfinite(first).any()
            else float("nan")
        ),
        "mean_distinct_targets": mean("distinct_targets"),
        "mean_concepts": mean("concepts"),
        "mean_false_expansion": mean("false_expansion"),
        "mean_reused_concepts": mean("reused_concepts"),
        "mean_useful_reuses": mean("useful_reuses"),
        "mean_max_target_depth": mean("max_target_depth"),
        "mean_positive_gain": mean("mean_positive_gain"),
        "mean_best_validation": mean("best_validation"),
        "mean_selected_heldout": mean("selected_heldout"),
        "mean_best_heldout": mean("best_heldout"),
        "mean_selection_regret": mean("selection_regret"),
    }
