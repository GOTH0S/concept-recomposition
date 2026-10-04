from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .archive import ConceptArchive
from .evaluator import score_expression
from .expression import Expression
from .grammar import OPS, concept_pool, operator_counts, proposal_pool, sample_pool
from .worlds import World

Arm = Literal["reset", "reify", "process", "sham"]


@dataclass(frozen=True)
class CandidateRecord:
    proposal: int
    generation: int
    expression: str
    expanded_key: str
    expanded_size: int
    expanded_depth: int
    root_op: str
    validation_score: float
    heldout_score: float
    validation_gain: float
    exact_target: bool
    target_key: str | None
    concept_refs: tuple[str, ...]


@dataclass
class SearchResult:
    arm: Arm
    records: list[CandidateRecord]
    archive: ConceptArchive
    first_target_proposal: int | None
    distinct_targets: tuple[str, ...]
    useful_concepts: tuple[str, ...]
    generation_ends: tuple[int, ...]
    promotion_schedule: tuple[tuple[int, ...], ...]


class SearchRunner:
    def __init__(
        self,
        world: World,
        arm: Arm,
        *,
        seed: int,
        proposal_budget: int,
        generation_size: int = 50,
        active_concepts: int = 4,
        promotions_per_generation: int = 2,
        promotion_threshold: float | None = None,
        split: int = 240,
        score_cache: dict[str, tuple[float, float]] | None = None,
        promotion_schedule: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        if proposal_budget < generation_size:
            raise ValueError("proposal budget must cover at least one generation")
        if proposal_budget % generation_size:
            raise ValueError("proposal budget must be a multiple of generation_size")
        if arm == "sham" and promotion_schedule is None:
            raise ValueError("sham requires the REIFY promotion schedule")

        self.world = world
        self.arm = arm
        self.rng = np.random.default_rng(seed)
        self.proposal_budget = proposal_budget
        self.generation_size = generation_size
        self.active_concepts = active_concepts
        self.promotions_per_generation = promotions_per_generation
        self.promotion_threshold = promotion_threshold
        self.split = split
        self.score_cache = score_cache if score_cache is not None else {}
        self.reference_schedule = promotion_schedule
        self.archive = ConceptArchive()
        self.op_weights = {op: 1.0 for op in OPS}
        self.eval_cache: dict[str, np.ndarray] = {}

    def _score(self, expression: Expression) -> tuple[float, float]:
        expanded = expression.expanded(self.archive.expressions)
        if expanded.key not in self.score_cache:
            self.score_cache[expanded.key] = score_expression(
                expanded,
                self.world,
                {},
                split=self.split,
                cache=self.eval_cache,
            )
        return self.score_cache[expanded.key]

    def _batch(self, raw_variables: tuple[str, ...]) -> tuple[Expression, ...]:
        base = proposal_pool(raw_variables)
        if self.arm == "reset" or not self.archive.ids:
            return sample_pool(
                self.rng,
                base,
                self.generation_size,
                self.op_weights if self.arm == "process" else None,
            )

        active = self.archive.active_ids(self.active_concepts)
        derived = concept_pool(raw_variables, active)
        derived_count = min(
            round(self.generation_size * 0.7),
            len(derived),
        )
        base_count = self.generation_size - derived_count
        return (
            sample_pool(
                self.rng,
                base,
                base_count,
                self.op_weights if self.arm == "process" else None,
            )
            + sample_pool(
                self.rng,
                derived,
                derived_count,
                self.op_weights if self.arm == "process" else None,
            )
        )

    def _promote_scored(
        self,
        candidates: list[tuple[Expression, Expression, float, float]],
        generation: int,
    ) -> tuple[int, ...]:
        ranked = sorted(candidates, key=lambda item: item[2], reverse=True)
        sizes: list[int] = []
        for expression, expanded, validation, _ in ranked:
            if (
                self.promotion_threshold is not None
                and validation < self.promotion_threshold
            ):
                break
            concept = self.archive.add(
                expression,
                expanded,
                validation,
                generation,
            )
            if concept is None:
                continue
            sizes.append(expanded.size)
            if self.arm == "process":
                for op, count in operator_counts(expression).items():
                    self.op_weights[op] += count * max(validation, 0.05)
            if len(sizes) == self.promotions_per_generation:
                break
        return tuple(sizes)

    def _promote_sham(
        self,
        candidates: list[tuple[Expression, Expression, float, float]],
        generation: int,
    ) -> tuple[int, ...]:
        assert self.reference_schedule is not None
        target_sizes = self.reference_schedule[generation]
        if not target_sizes:
            return ()

        ranked = sorted(candidates, key=lambda item: item[2])
        tail = ranked[: max(len(ranked) // 2, len(target_sizes))]
        used: set[str] = set()
        promoted: list[int] = []

        for target_size in target_sizes:
            choices = [
                item
                for item in tail
                if item[1].key not in used
            ]
            if not choices:
                break
            expression, expanded, validation, _ = min(
                choices,
                key=lambda item: (
                    abs(item[1].size - target_size),
                    item[2],
                ),
            )
            concept = self.archive.add(
                expression,
                expanded,
                validation,
                generation,
            )
            if concept is None:
                used.add(expanded.key)
                continue
            used.add(expanded.key)
            promoted.append(expanded.size)

        return tuple(promoted)

    def run(self) -> SearchResult:
        records: list[CandidateRecord] = []
        first_target: int | None = None
        distinct_targets: set[str] = set()
        target_concepts: set[str] = set()
        generation_ends: list[int] = []
        promotion_schedule: list[tuple[int, ...]] = []

        raw_variables = tuple(self.world.data)
        generations = self.proposal_budget // self.generation_size
        proposal = 0

        for generation in range(generations):
            candidates: list[tuple[Expression, Expression, float, float]] = []
            for expression in self._batch(raw_variables):
                validation, heldout = self._score(expression)
                expanded = expression.expanded(self.archive.expressions)
                refs = expression.concept_ids()
                parent_best = max(
                    (self.archive.score(ref) for ref in refs),
                    default=validation,
                )
                gain = validation - parent_best if refs else 0.0
                exact = expanded.key in self.world.target_keys

                proposal += 1
                if refs:
                    self.archive.note_use(
                        refs,
                        useful=(
                            self.promotion_threshold is None
                            or validation >= self.promotion_threshold
                        ),
                    )
                if exact:
                    distinct_targets.add(expanded.key)
                    target_concepts.update(refs)
                    if first_target is None:
                        first_target = proposal

                records.append(
                    CandidateRecord(
                        proposal=proposal,
                        generation=generation,
                        expression=str(expression),
                        expanded_key=expanded.key,
                        expanded_size=expanded.size,
                        expanded_depth=expanded.depth,
                        root_op=expanded.op,
                        validation_score=validation,
                        heldout_score=heldout,
                        validation_gain=gain,
                        exact_target=exact,
                        target_key=expanded.key if exact else None,
                        concept_refs=refs,
                    )
                )
                candidates.append(
                    (expression, expanded, validation, heldout)
                )

            if len(candidates) != self.generation_size:
                raise RuntimeError("proposal pool could not fill the generation")

            if self.arm == "reset":
                promoted = ()
            elif self.arm == "sham":
                promoted = self._promote_sham(candidates, generation)
            else:
                promoted = self._promote_scored(candidates, generation)

            promotion_schedule.append(promoted)
            generation_ends.append(proposal)

        useful = self.archive.lineage(tuple(target_concepts)) if target_concepts else set()
        return SearchResult(
            arm=self.arm,
            records=records,
            archive=self.archive,
            first_target_proposal=first_target,
            distinct_targets=tuple(sorted(distinct_targets)),
            useful_concepts=tuple(sorted(useful)),
            generation_ends=tuple(generation_ends),
            promotion_schedule=tuple(promotion_schedule),
        )
