from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .archive import ConceptArchive
from .evaluator import score_expression
from .grammar import Grammar
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
        local_depth: int = 2,
    ) -> None:
        if arm != "reset":
            raise NotImplementedError("reification is added in the next unit")
        if proposal_budget < 1:
            raise ValueError("proposal_budget must be positive")
        if generation_size < 1:
            raise ValueError("generation_size must be positive")

        self.world = world
        self.arm = arm
        self.rng = np.random.default_rng(seed)
        self.proposal_budget = proposal_budget
        self.generation_size = generation_size
        self.grammar = Grammar(tuple(world.data), max_local_depth=local_depth)
        self.eval_cache: dict[str, np.ndarray] = {}

    def run(self) -> SearchResult:
        records: list[CandidateRecord] = []
        first_target: int | None = None
        distinct_targets: set[str] = set()
        generation_ends: list[int] = []

        for proposal in range(1, self.proposal_budget + 1):
            expression = self.grammar.sample(self.rng)
            validation, heldout = score_expression(
                expression,
                self.world,
                {},
                cache=self.eval_cache,
            )
            exact = expression.key in self.world.target_keys
            if exact:
                distinct_targets.add(expression.key)
                if first_target is None:
                    first_target = proposal

            generation = (proposal - 1) // self.generation_size
            records.append(
                CandidateRecord(
                    proposal=proposal,
                    generation=generation,
                    expression=str(expression),
                    expanded_key=expression.key,
                    expanded_size=expression.size,
                    expanded_depth=expression.depth,
                    root_op=expression.op,
                    validation_score=validation,
                    heldout_score=heldout,
                    validation_gain=0.0,
                    exact_target=exact,
                    target_key=expression.key if exact else None,
                    concept_refs=(),
                )
            )
            if proposal % self.generation_size == 0 or proposal == self.proposal_budget:
                generation_ends.append(proposal)

        return SearchResult(
            arm="reset",
            records=records,
            archive=ConceptArchive(),
            first_target_proposal=first_target,
            distinct_targets=tuple(sorted(distinct_targets)),
            useful_concepts=(),
            generation_ends=tuple(generation_ends),
            promotion_schedule=tuple(() for _ in generation_ends),
        )
