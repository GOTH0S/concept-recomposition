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


def _target_lineages(
    result: SearchResult,
    budget: int,
) -> list[tuple[str, set[str]]]:
    lineages: list[tuple[str, set[str]]] = []
    for record in result.records:
        if (
            record.proposal > budget
            or record.target_key is None
        ):
            continue
        lineages.append(
            (
                record.target_key,
                result.archive.lineage(record.concept_refs),
            )
        )
    return lineages


def _prefix_metrics(result: SearchResult, budget: int) -> dict[str, float]:
    records = [record for record in result.records if record.proposal <= budget]
    matched = [record for record in records if record.target_key is not None]
    available = _available_concepts(result, budget)
    target_lineages = _target_lineages(result, budget)
    useful = (
        set().union(*(lineage for _, lineage in target_lineages))
        if target_lineages
        else set()
    )

    target_use: dict[str, set[str]] = {}
    for target_key, lineage in target_lineages:
        for concept_id in lineage:
            target_use.setdefault(concept_id, set()).add(target_key)
    shared = {
        concept_id
        for concept_id, targets in target_use.items()
        if len(targets) >= 2
    }

    selected = max(records, key=lambda record: record.promotion_score)
    best_heldout = max(record.heldout_score for record in records)
    concepts = [
        row
        for row in result.archive.rows()
        if str(row["concept_id"]) in available
    ]
    relevant = [
        row
        for row in concepts
        if str(row["expanded_key"]) in result.target_subexpressions
    ]

    distinct = len({record.target_key for record in matched})
    reached = (
        result.required_targets > 0
        and distinct >= result.required_targets
    )

    return {
        "reached_target": float(reached),
        "first_target": float(
            min(record.proposal for record in matched)
        ) if matched else float("nan"),
        "first_success": float(
            max(
                min(record.proposal for record in matched if record.target_key == key)
                for key in {record.target_key for record in matched}
            )
        ) if reached else float("nan"),
        "distinct_targets": float(distinct),
        "concepts": float(len(available)),
        "unused_promotion_rate": (
            0.0 if not available else 1.0 - len(useful & available) / len(available)
        ),
        "relevant_promotions": float(len(relevant)),
        "relevant_promotion": float(bool(relevant)),
        "promotion_precision": (
            0.0 if not concepts else len(relevant) / len(concepts)
        ),
        "reused_concepts": float(
            sum(int(row["proposal_uses"]) > 0 for row in concepts)
        ),
        "shared_target_concepts": float(len(shared & available)),
        "useful_reuses": float(
            sum(int(row["useful_uses"]) for row in concepts)
        ),
        "max_target_depth": float(
            max((record.expanded_depth for record in matched), default=0)
        ),
        "max_target_compression": float(
            max(
                (
                    record.expanded_size - record.local_size
                    for record in matched
                ),
                default=0,
            )
        ),
        "mean_positive_gain": float(
            np.mean(
                [
                    record.confirmation_gain
                    for record in records
                    if record.concept_refs and record.confirmation_gain > 0
                ]
            )
        ) if any(
            record.concept_refs and record.confirmation_gain > 0
            for record in records
        ) else 0.0,
        "best_promotion": selected.promotion_score,
        "selected_heldout": selected.heldout_score,
        "best_heldout": best_heldout,
        "selection_regret": best_heldout - selected.heldout_score,
    }


def summarize(
    results: Iterable[SearchResult],
    budget: int,
) -> dict[str, float]:
    values = [_prefix_metrics(result, budget) for result in results]
    if not values:
        raise ValueError("results cannot be empty")

    first = np.array([value["first_success"] for value in values], dtype=float)

    def mean(field: str) -> float:
        return float(np.mean([value[field] for value in values]))

    return {
        "reach_rate": mean("reached_target"),
        "mean_first_success": (
            float(np.nanmean(first))
            if np.isfinite(first).any()
            else float("nan")
        ),
        "mean_distinct_targets": mean("distinct_targets"),
        "mean_concepts": mean("concepts"),
        "mean_unused_promotion_rate": mean("unused_promotion_rate"),
        "mean_relevant_promotions": mean("relevant_promotions"),
        "relevant_promotion_rate": mean("relevant_promotion"),
        "mean_promotion_precision": mean("promotion_precision"),
        "mean_reused_concepts": mean("reused_concepts"),
        "mean_shared_target_concepts": mean("shared_target_concepts"),
        "mean_useful_reuses": mean("useful_reuses"),
        "mean_max_target_depth": mean("max_target_depth"),
        "mean_max_target_compression": mean("max_target_compression"),
        "mean_positive_gain": mean("mean_positive_gain"),
        "mean_best_promotion": mean("best_promotion"),
        "mean_selected_heldout": mean("selected_heldout"),
        "mean_best_heldout": mean("best_heldout"),
        "mean_selection_regret": mean("selection_regret"),
    }
