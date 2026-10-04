from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .archive import ConceptArchive
from .expression import Expression
from .metrics import score
from .operators import evaluate
from .proposer import ProposalPolicy, local_pool, sample_pool
from .worlds import World

Arm = Literal["reset", "reify", "process", "sham"]


@dataclass(frozen=True, slots=True)
class SearchConfig:
    proposals: int = 640
    generation_size: int = 80
    promotions_per_generation: int = 2
    active_concepts: int = 4


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    attempt: int
    generation: int
    expression: str
    expanded_expression: str
    local_depth: int
    expanded_depth: int
    validation_score: float
    test_score: float
    reached_target: bool


@dataclass(slots=True)
class SearchResult:
    arm: Arm
    world: str
    seed: int
    first_reach_attempt: int | None
    concepts_promoted: int
    concepts_reused: int
    max_useful_depth: int
    false_expansion_rate: float
    records: list[CandidateRecord]


class SearchRunner:
    def __init__(
        self,
        world: World,
        arm: Arm,
        config: SearchConfig,
        seed: int,
    ):
        self.world = world
        self.arm = arm
        self.config = config
        self.seed = seed

    def run(self) -> SearchResult:
        rng = np.random.default_rng(self.seed)
        archive = ConceptArchive()
        policy = ProposalPolicy()
        raw_terminals = tuple(
            Expression.var(name)
            for name in sorted(self.world.variables)
        )
        split = int(len(self.world.y) * 0.7)
        value_cache: dict[Expression, np.ndarray] = {}
        promoted_expanded: set[str] = set()
        useful_concepts: set[str] = set()
        records: list[CandidateRecord] = []
        first_reach: int | None = None
        max_useful_depth = 0

        base_pool = local_pool(raw_terminals, raw_terminals)
        generations = (
            self.config.proposals // self.config.generation_size
        )
        for generation in range(generations):
            terminals = raw_terminals
            if self.arm != "reset":
                terminals += archive.terminals(
                    self.config.active_concepts
                )
            concept_values = archive.values(
                self.world.variables,
                value_cache,
            )

            proposals = self._proposals(
                rng,
                policy,
                base_pool,
                raw_terminals,
                terminals,
                generation,
            )

            batch: list[
                tuple[Expression, float, float, bool]
            ] = []
            for offset, expression in enumerate(proposals, start=1):
                signal = evaluate(
                    expression,
                    self.world.variables,
                    concept_values,
                    value_cache,
                )
                validation, test = score(
                    signal,
                    self.world.y,
                    split,
                )
                expanded = archive.expand(expression)
                reached = (
                    self.world.target is not None
                    and expanded == self.world.target
                )
                attempt_number = (
                    generation * self.config.generation_size
                    + offset
                )
                records.append(
                    CandidateRecord(
                        attempt_number,
                        generation,
                        str(expression),
                        str(expanded),
                        expression.local_depth,
                        expanded.local_depth,
                        validation,
                        test,
                        reached,
                    )
                )
                batch.append(
                    (expression, validation, test, reached)
                )
                archive.mark_reuse(expression)
                if reached and first_reach is None:
                    first_reach = attempt_number
                if reached:
                    max_useful_depth = max(
                        max_useful_depth,
                        expanded.local_depth,
                    )
                    useful_concepts.update(
                        archive.lineage_concepts(expression)
                    )

            if self.arm == "reset":
                continue

            selected = self._promotions(
                rng,
                archive,
                batch,
                promoted_expanded,
            )
            promoted: list[Expression] = []
            for expression, validation, _, _ in selected:
                expanded_key = str(archive.expand(expression))
                archive.add(
                    expression,
                    generation,
                    validation,
                )
                promoted_expanded.add(expanded_key)
                promoted.append(expression)
            if self.arm == "process":
                policy.update(promoted)

        concepts = list(archive.concepts.values())
        reused = sum(
            concept.reuse_count > 0
            for concept in concepts
        )
        false_rate = (
            0.0
            if not concepts
            else 1.0 - len(useful_concepts) / len(concepts)
        )
        return SearchResult(
            self.arm,
            self.world.name,
            self.seed,
            first_reach,
            len(concepts),
            reused,
            max_useful_depth,
            false_rate,
            records,
        )

    def _proposals(
        self,
        rng: np.random.Generator,
        policy: ProposalPolicy,
        base_pool: tuple[Expression, ...],
        raw_terminals: tuple[Expression, ...],
        terminals: tuple[Expression, ...],
        generation: int,
    ) -> tuple[Expression, ...]:
        if generation == 0:
            unary = tuple(
                expr
                for expr in base_pool
                if expr.kind == "unary"
            )
            other = tuple(
                expr
                for expr in base_pool
                if expr.kind != "unary"
            )
            extra = sample_pool(
                rng,
                other,
                policy,
                min(
                    self.config.generation_size - len(unary),
                    len(other),
                ),
            )
            return unary + extra

        if self.arm == "reset":
            return sample_pool(
                rng,
                base_pool,
                policy,
                min(
                    self.config.generation_size,
                    len(base_pool),
                ),
            )

        expanded_pool = local_pool(
            terminals,
            raw_terminals,
        )
        concept_pool = tuple(
            expr
            for expr in expanded_pool
            if any(
                arg.kind == "concept"
                for arg in expr.args
            )
        )
        base_count = min(20, len(base_pool))
        concept_count = min(
            self.config.generation_size - base_count,
            len(concept_pool),
        )
        return (
            sample_pool(
                rng,
                base_pool,
                policy,
                base_count,
            )
            + sample_pool(
                rng,
                concept_pool,
                policy,
                concept_count,
            )
        )

    def _promotions(
        self,
        rng: np.random.Generator,
        archive: ConceptArchive,
        batch: list[
            tuple[Expression, float, float, bool]
        ],
        promoted_expanded: set[str],
    ) -> list[
        tuple[Expression, float, float, bool]
    ]:
        promotable = []
        batch_seen: set[str] = set()
        for item in batch:
            expression = item[0]
            expanded_key = str(archive.expand(expression))
            if (
                expanded_key in promoted_expanded
                or expanded_key in batch_seen
            ):
                continue
            batch_seen.add(expanded_key)
            promotable.append(item)

        ranked = sorted(
            promotable,
            key=lambda item: item[1],
            reverse=True,
        )
        k = min(
            self.config.promotions_per_generation,
            len(ranked),
        )
        if self.arm != "sham":
            return ranked[:k]

        tail = ranked[len(ranked) // 2 :]
        if not tail:
            return []
        return list(
            rng.choice(
                tail,
                size=min(k, len(tail)),
                replace=False,
            )
        )
