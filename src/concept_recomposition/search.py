from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .archive import ConceptArchive
from .evaluator import score_expression
from .expression import Expression
from .grammar import OPS, base_templates, concept_proposals, operator_counts, random_expression
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
    exact_target: bool
    target_key: str | None
    concept_refs: tuple[str, ...]


@dataclass
class SearchResult:
    arm: Arm
    records: list[CandidateRecord]
    archive: ConceptArchive
    first_target_proposal: int | None
    target_hits: int
    distinct_targets: tuple[str, ...]
    useful_concepts: tuple[str, ...]
    best_validation: float
    best_heldout: float
    sham_schedule: tuple[tuple[int, ...], ...]

    @property
    def reached_target(self) -> bool:
        return self.first_target_proposal is not None


class SearchRunner:
    def __init__(
        self,
        world: World,
        arm: Arm,
        *,
        seed: int,
        proposal_budget: int,
        generations: int = 10,
        max_size: int = 7,
        promotions_per_generation: int = 4,
        promotion_threshold: float = 0.22,
        split: int = 240,
        sham_schedule: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        if proposal_budget < generations:
            raise ValueError("proposal budget must be at least the number of generations")
        if arm == "sham" and sham_schedule is None:
            raise ValueError("sham requires a promotion schedule")
        self.world = world
        self.arm = arm
        self.rng = np.random.default_rng(seed)
        self.proposal_budget = proposal_budget
        self.generations = generations
        self.max_size = max_size
        self.promotions_per_generation = promotions_per_generation
        self.promotion_threshold = promotion_threshold
        self.split = split
        self.sham_schedule = sham_schedule
        self.archive = ConceptArchive()
        self.op_weights = {op: 1.0 for op in OPS}
        self.eval_cache: dict[str, np.ndarray] = {}

    def _expanded_key(self, expression: Expression) -> str:
        return expression.expanded(self.archive.expressions).key


    def _promote(
        self,
        candidates: list[tuple[Expression, float, float]],
        generation: int,
    ) -> tuple[int, ...]:
        if self.arm == "reset":
            return ()

        promoted_sizes: list[int] = []
        if self.arm in {"reify", "process"}:
            ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
            for expression, validation, _ in ranked:
                if validation < self.promotion_threshold:
                    break
                if expression.size <= 1:
                    continue
                concept = self.archive.add(expression, validation, generation)
                if concept is None:
                    continue
                promoted_sizes.append(expression.size)
                if self.arm == "process":
                    for op, count in operator_counts(expression).items():
                        self.op_weights[op] += count * max(validation, 0.05)
                if len(promoted_sizes) >= self.promotions_per_generation:
                    break
            return tuple(promoted_sizes)

        targets = self.sham_schedule[generation]
        if not targets:
            return ()
        for target_size in targets:
            choices: list[tuple[Expression, float]] = []
            for _ in range(24):
                expression = random_expression(
                    self.rng,
                    tuple(self.world.data),
                    (),
                    max(2, target_size),
                )
                validation, _ = score_expression(
                    expression, self.world, {}, split=self.split
                )
                if expression.size > 1:
                    choices.append((expression, validation))
            if not choices:
                continue
            expression, validation = min(
                choices,
                key=lambda item: (
                    item[1] >= self.promotion_threshold,
                    abs(item[0].size - target_size),
                    item[1],
                ),
            )
            concept = self.archive.add(expression, validation, generation)
            if concept is not None:
                promoted_sizes.append(expression.size)
        return tuple(promoted_sizes)

    def run(self) -> SearchResult:
        records: list[CandidateRecord] = []
        first_target: int | None = None
        target_hits = 0
        distinct_targets: set[str] = set()
        target_concepts: set[str] = set()
        best_validation = -np.inf
        best_heldout = -np.inf
        promotion_schedule: list[tuple[int, ...]] = []

        raw_variables = tuple(self.world.data)
        base, remainder = divmod(self.proposal_budget, self.generations)
        proposal = 0
        base_pool = base_templates(raw_variables)
        self.rng.shuffle(base_pool)
        base_cursor = 0

        for generation in range(self.generations):
            count = base + (1 if generation < remainder else 0)
            generation_candidates: list[tuple[Expression, float, float]] = []
            seen: set[str] = set()
            concepts = self.archive.ids if self.arm != "reset" else ()
            concept_count = min(len(concepts) * 2, count // 2)
            structured = concept_proposals(
                self.rng,
                raw_variables,
                concepts,
                concept_count,
                self.op_weights if self.arm == "process" else None,
            )
            room = max(0, count - len(structured))
            if room:
                structured.extend(base_pool[base_cursor : base_cursor + room])
                base_cursor += room

            for slot in range(count):
                if slot < len(structured):
                    expression = structured[slot]
                else:
                    expression = random_expression(
                        self.rng,
                        raw_variables,
                        concepts,
                        self.max_size,
                        self.op_weights if self.arm == "process" else None,
                    )
                if expression.key in seen:
                    proposal += 1
                    continue
                seen.add(expression.key)
                validation, heldout = score_expression(
                    expression,
                    self.world,
                    self.archive.expressions,
                    split=self.split,
                    cache=self.eval_cache,
                )
                refs = expression.concept_ids()
                useful = validation >= self.promotion_threshold
                if refs:
                    self.archive.note_use(refs, useful)
                expanded = expression.expanded(self.archive.expressions)
                expanded_key = expanded.key
                exact = expanded_key in self.world.target_keys
                target_key = expanded_key if exact else None
                if exact:
                    target_hits += 1
                    distinct_targets.add(expanded_key)
                    target_concepts.update(refs)
                    if first_target is None:
                        first_target = proposal + 1
                best_validation = max(best_validation, validation)
                best_heldout = max(best_heldout, heldout)
                records.append(
                    CandidateRecord(
                        proposal=proposal + 1,
                        generation=generation,
                        expression=str(expression),
                        expanded_key=expanded_key,
                        expanded_size=expanded.size,
                        expanded_depth=expanded.depth,
                        root_op=expanded.op,
                        validation_score=validation,
                        heldout_score=heldout,
                        exact_target=exact,
                        target_key=target_key,
                        concept_refs=refs,
                    )
                )
                generation_candidates.append((expression, validation, heldout))
                proposal += 1

            promotion_schedule.append(self._promote(generation_candidates, generation))

        return SearchResult(
            arm=self.arm,
            records=records,
            archive=self.archive,
            first_target_proposal=first_target,
            target_hits=target_hits,
            distinct_targets=tuple(sorted(distinct_targets)),
            useful_concepts=tuple(sorted(self.archive.lineage(tuple(target_concepts)))),
            best_validation=float(best_validation),
            best_heldout=float(best_heldout),
            sham_schedule=tuple(promotion_schedule),
        )
