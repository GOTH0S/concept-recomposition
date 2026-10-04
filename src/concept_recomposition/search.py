from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .archive import ConceptArchive
from .evaluator import EvidenceScore, score_expression
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
    discovery_score: float
    confirmation_score: float
    promotion_score: float
    heldout_score: float
    confirmation_gain: float
    exact_target: bool
    target_key: str | None
    concept_refs: tuple[str, ...]


@dataclass
class SearchResult:
    world: str
    arm: Arm
    records: list[CandidateRecord]
    archive: ConceptArchive
    first_target_proposal: int | None
    first_success_proposal: int | None
    required_targets: int
    distinct_targets: tuple[str, ...]
    useful_concepts: tuple[str, ...]
    target_subexpressions: frozenset[str]
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
        promotions_per_generation: int = 1,
        promotion_threshold: float = 0.20,
        discovery_end: int = 180,
        heldout_start: int = 270,
        score_cache: dict[str, EvidenceScore] | None = None,
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
        self.discovery_end = discovery_end
        self.heldout_start = heldout_start
        self.score_cache = score_cache if score_cache is not None else {}
        self.reference_schedule = promotion_schedule
        self.archive = ConceptArchive()
        self.op_weights = {op: 1.0 for op in OPS}
        self.eval_cache: dict[str, np.ndarray] = {}

    def _score(self, expression: Expression) -> EvidenceScore:
        expanded = expression.expanded(self.archive.expressions)
        if expanded.key not in self.score_cache:
            self.score_cache[expanded.key] = score_expression(
                expanded,
                self.world,
                {},
                discovery_end=self.discovery_end,
                heldout_start=self.heldout_start,
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
        derived_count = min(round(self.generation_size * 0.7), len(derived))
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
        candidates: list[tuple[Expression, Expression, EvidenceScore]],
        generation: int,
    ) -> tuple[int, ...]:
        ranked = sorted(
            candidates,
            key=lambda item: item[2].promotion,
            reverse=True,
        )
        sizes: list[int] = []
        for expression, expanded, score in ranked:
            if score.promotion < self.promotion_threshold:
                break
            concept = self.archive.add(
                expression,
                expanded,
                score.promotion,
                generation,
            )
            if concept is None:
                continue
            sizes.append(expanded.size)
            if self.arm == "process":
                for op, count in operator_counts(expression).items():
                    self.op_weights[op] += count * score.promotion
            if len(sizes) == self.promotions_per_generation:
                break
        return tuple(sizes)

    def _promote_sham(
        self,
        candidates: list[tuple[Expression, Expression, EvidenceScore]],
        generation: int,
    ) -> tuple[int, ...]:
        assert self.reference_schedule is not None
        target_sizes = self.reference_schedule[generation]
        if not target_sizes:
            return ()

        ranked = sorted(candidates, key=lambda item: item[2].promotion)
        tail = ranked[: max(len(ranked) // 2, len(target_sizes))]
        used: set[str] = set()
        promoted: list[int] = []

        for target_size in target_sizes:
            choices = [item for item in tail if item[1].key not in used]
            while choices:
                expression, expanded, score = min(
                    choices,
                    key=lambda item: (
                        abs(item[1].size - target_size),
                        item[2].promotion,
                    ),
                )
                used.add(expanded.key)
                concept = self.archive.add(
                    expression,
                    expanded,
                    score.promotion,
                    generation,
                )
                if concept is not None:
                    promoted.append(expanded.size)
                    break
                choices = [
                    item
                    for item in choices
                    if item[1].key not in used
                ]

        return tuple(promoted)

    def run(self) -> SearchResult:
        records: list[CandidateRecord] = []
        first_target: int | None = None
        first_success: int | None = None
        distinct_targets: set[str] = set()
        target_concepts: set[str] = set()
        generation_ends: list[int] = []
        promotion_schedule: list[tuple[int, ...]] = []

        raw_variables = tuple(self.world.data)
        generations = self.proposal_budget // self.generation_size
        proposal = 0

        for generation in range(generations):
            candidates: list[tuple[Expression, Expression, EvidenceScore]] = []
            for expression in self._batch(raw_variables):
                score = self._score(expression)
                expanded = expression.expanded(self.archive.expressions)
                refs = expression.concept_ids()
                parent_best = max(
                    (self.archive.score(ref) for ref in refs),
                    default=score.promotion,
                )
                gain = score.confirmation - parent_best if refs else 0.0
                exact = expanded.key in self.world.target_keys

                proposal += 1
                if refs:
                    self.archive.note_use(refs, useful=gain > 0)
                if exact:
                    distinct_targets.add(expanded.key)
                    target_concepts.update(refs)
                    if first_target is None:
                        first_target = proposal
                    if (
                        first_success is None
                        and len(distinct_targets) >= self.world.required_targets
                    ):
                        first_success = proposal

                records.append(
                    CandidateRecord(
                        proposal=proposal,
                        generation=generation,
                        expression=str(expression),
                        expanded_key=expanded.key,
                        expanded_size=expanded.size,
                        expanded_depth=expanded.depth,
                        root_op=expanded.op,
                        discovery_score=score.discovery,
                        confirmation_score=score.confirmation,
                        promotion_score=score.promotion,
                        heldout_score=score.heldout,
                        confirmation_gain=gain,
                        exact_target=exact,
                        target_key=expanded.key if exact else None,
                        concept_refs=refs,
                    )
                )
                candidates.append((expression, expanded, score))

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
        target_subexpressions = frozenset(
            key
            for target in self.world.targets
            for key in target.subexpression_keys()
            if not key.startswith("raw:") and key != target.key
        )
        return SearchResult(
            world=self.world.name,
            arm=self.arm,
            records=records,
            archive=self.archive,
            first_target_proposal=first_target,
            first_success_proposal=first_success,
            required_targets=self.world.required_targets,
            distinct_targets=tuple(sorted(distinct_targets)),
            useful_concepts=tuple(sorted(useful)),
            target_subexpressions=target_subexpressions,
            generation_ends=tuple(generation_ends),
            promotion_schedule=tuple(promotion_schedule),
        )
