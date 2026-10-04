from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .archive import ConceptArchive
from .evaluator import score_expression
from .expression import Expression
from .grammar import Grammar
from .proposer import MotifModel
from .worlds import World

Arm = Literal["reset", "reify", "process", "sham"]


@dataclass(frozen=True)
class PromotionShape:
    size: int
    depth: int


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
    novel: bool = True
    exact_intermediate: bool = False


@dataclass
class SearchResult:
    arm: Arm
    records: list[CandidateRecord]
    archive: ConceptArchive
    first_target_proposal: int | None
    distinct_targets: tuple[str, ...]
    useful_concepts: tuple[str, ...]
    generation_ends: tuple[int, ...]
    promotion_schedule: tuple[tuple[PromotionShape, ...], ...]
    promoted_intermediates: tuple[str, ...] = ()


class SearchRunner:
    def __init__(
        self,
        world: World,
        arm: Arm,
        *,
        seed: int,
        proposal_budget: int,
        generation_size: int = 50,
        local_depth: int = 2,
        active_concepts: int = 3,
        promotions_per_generation: int = 1,
        promotion_threshold: float = 0.20,
        promotion_schedule: tuple[tuple[PromotionShape, ...], ...] | None = None,
        score_cache: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if arm == "sham" and promotion_schedule is None:
            raise ValueError("sham requires a REIFY promotion schedule")
        if proposal_budget < 1 or generation_size < 1:
            raise ValueError("proposal_budget and generation_size must be positive")

        self.world = world
        self.arm = arm
        self.rng = np.random.default_rng(seed)
        self.proposal_budget = proposal_budget
        self.generation_size = generation_size
        self.active_concepts = active_concepts
        self.promotions_per_generation = promotions_per_generation
        self.promotion_threshold = promotion_threshold
        self.reference_schedule = promotion_schedule
        self.grammar = Grammar(
            tuple(world.data),
            max_local_depth=local_depth,
            terminal_probability=world.terminal_probability,
            operators=world.grammar_ops,
            param_choices=dict(world.grammar_params),
            conditions=world.conditions,
        )
        self.archive = ConceptArchive()
        self.motifs = MotifModel()
        self.score_cache = score_cache if score_cache is not None else {}
        self.eval_cache: dict[str, np.ndarray] = {}

    def _score(self, expanded: Expression) -> tuple[float, float]:
        if expanded.key not in self.score_cache:
            self.score_cache[expanded.key] = score_expression(
                expanded,
                self.world,
                {},
                cache=self.eval_cache,
            )
        return self.score_cache[expanded.key]

    def _promote_reify(
        self,
        candidates: list[tuple[Expression, Expression, float, int, bool]],
        generation: int,
    ) -> tuple[PromotionShape, ...]:
        promoted: list[PromotionShape] = []
        ranked = sorted(candidates, key=lambda item: item[2], reverse=True)
        for expression, expanded, validation, proposal, novel in ranked:
            if validation < self.promotion_threshold:
                break
            if not novel:
                continue
            concept = self.archive.add(
                expression,
                expanded,
                validation,
                generation,
                proposal,
            )
            if concept is None:
                continue
            promoted.append(PromotionShape(concept.expanded_size, concept.expanded_depth))
            if self.arm == "process":
                self.motifs.observe(expression, validation)
            if len(promoted) == self.promotions_per_generation:
                break
        return tuple(promoted)

    def _promote_sham(
        self,
        candidates: list[tuple[Expression, Expression, float, int, bool]],
        generation: int,
    ) -> tuple[PromotionShape, ...]:
        assert self.reference_schedule is not None
        if generation >= len(self.reference_schedule):
            return ()
        targets = self.reference_schedule[generation]
        if not targets:
            return ()

        novel = [item for item in candidates if item[4]]
        median = float(np.median([item[2] for item in novel])) if novel else 0.0
        pool = [item for item in novel if item[2] <= median]
        used: set[str] = set()
        promoted: list[PromotionShape] = []

        for target in targets:
            choices = [item for item in pool if item[1].key not in used]
            if not choices:
                choices = [item for item in novel if item[1].key not in used]
            if not choices:
                break
            expression, expanded, validation, proposal, _ = min(
                choices,
                key=lambda item: (
                    abs(item[1].depth - target.depth),
                    abs(item[1].size - target.size),
                    item[2],
                ),
            )
            used.add(expanded.key)
            concept = self.archive.add(
                expression,
                expanded,
                validation,
                generation,
                proposal,
            )
            if concept is not None:
                promoted.append(PromotionShape(concept.expanded_size, concept.expanded_depth))
        return tuple(promoted)

    def run(self) -> SearchResult:
        records: list[CandidateRecord] = []
        first_target: int | None = None
        distinct_targets: set[str] = set()
        target_concepts: set[str] = set()
        generation_ends: list[int] = []
        promotion_schedule: list[tuple[PromotionShape, ...]] = []
        seen: set[str] = set()
        intermediate_keys = {expression.key for expression in self.world.intermediates}

        generations = (self.proposal_budget + self.generation_size - 1) // self.generation_size
        proposal = 0

        for generation in range(generations):
            concept_ids = (
                ()
                if self.arm == "reset"
                else self.archive.active_ids(self.active_concepts)
            )
            generation_candidates: list[
                tuple[Expression, Expression, float, int, bool]
            ] = []
            remaining = min(self.generation_size, self.proposal_budget - proposal)
            transition_weights = self.motifs.weights if self.arm == "process" else None

            for _ in range(remaining):
                expression = self.grammar.sample(
                    self.rng,
                    concept_ids=concept_ids,
                    transition_weights=transition_weights,
                )
                expanded = expression.expanded(self.archive.expressions)
                validation, heldout = self._score(expanded)
                refs = expression.concept_ids()
                parent_best = max(
                    (self.archive.score(ref) for ref in refs),
                    default=validation,
                )
                gain = validation - parent_best if refs else 0.0
                exact = expanded.key in self.world.target_keys
                exact_intermediate = expanded.key in intermediate_keys
                novel = expanded.key not in seen
                seen.add(expanded.key)

                proposal += 1
                if refs:
                    self.archive.note_use(
                        refs,
                        useful=gain > 0.02,
                        target=exact,
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
                        novel=novel,
                        exact_intermediate=exact_intermediate,
                    )
                )
                generation_candidates.append(
                    (expression, expanded, validation, proposal, novel)
                )

            if self.arm in {"reify", "process"}:
                promoted = self._promote_reify(generation_candidates, generation)
            elif self.arm == "sham":
                promoted = self._promote_sham(generation_candidates, generation)
            else:
                promoted = ()

            promotion_schedule.append(promoted)
            generation_ends.append(proposal)

        useful = self.archive.lineage(tuple(target_concepts)) if target_concepts else set()
        promoted_intermediates = tuple(
            sorted(
                str(row["expanded_key"])
                for row in self.archive.rows()
                if str(row["expanded_key"]) in intermediate_keys
            )
        )
        return SearchResult(
            arm=self.arm,
            records=records,
            archive=self.archive,
            first_target_proposal=first_target,
            distinct_targets=tuple(sorted(distinct_targets)),
            useful_concepts=tuple(sorted(useful)),
            generation_ends=tuple(generation_ends),
            promotion_schedule=tuple(promotion_schedule),
            promoted_intermediates=promoted_intermediates,
        )
