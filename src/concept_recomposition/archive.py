from __future__ import annotations

from dataclasses import dataclass

from .expression import Expression


@dataclass
class Concept:
    concept_id: str
    expression: Expression
    expanded_key: str
    expanded_size: int
    promotion_score: float
    generation: int
    proposal_uses: int = 0
    useful_uses: int = 0


class ConceptArchive:
    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {}
        self._by_expanded: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._concepts)

    @property
    def expressions(self) -> dict[str, Expression]:
        return {
            concept_id: concept.expression
            for concept_id, concept in self._concepts.items()
        }

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._concepts)

    def active_ids(self, limit: int) -> tuple[str, ...]:
        ranked = sorted(
            self._concepts.values(),
            key=lambda concept: (
                concept.promotion_score,
                -concept.generation,
            ),
            reverse=True,
        )
        return tuple(concept.concept_id for concept in ranked[:limit])

    def score(self, concept_id: str) -> float:
        return self._concepts[concept_id].promotion_score

    def add(
        self,
        expression: Expression,
        expanded: Expression,
        score: float,
        generation: int,
    ) -> Concept | None:
        if expanded.key in self._by_expanded:
            return None
        concept_id = f"C{len(self._concepts):03d}"
        concept = Concept(
            concept_id=concept_id,
            expression=expression,
            expanded_key=expanded.key,
            expanded_size=expanded.size,
            promotion_score=score,
            generation=generation,
        )
        self._concepts[concept_id] = concept
        self._by_expanded[expanded.key] = concept_id
        return concept

    def note_use(self, concept_ids: tuple[str, ...], useful: bool) -> None:
        for concept_id in set(concept_ids):
            concept = self._concepts[concept_id]
            concept.proposal_uses += 1
            if useful:
                concept.useful_uses += 1

    def lineage(self, concept_ids: tuple[str, ...]) -> set[str]:
        found = set(concept_ids)
        pending = list(concept_ids)
        while pending:
            concept_id = pending.pop()
            for parent in self._concepts[concept_id].expression.concept_ids():
                if parent not in found:
                    found.add(parent)
                    pending.append(parent)
        return found

    def rows(self) -> list[dict[str, object]]:
        return [
            {
                "concept_id": concept.concept_id,
                "expression": str(concept.expression),
                "expanded_key": concept.expanded_key,
                "expanded_size": concept.expanded_size,
                "promotion_score": concept.promotion_score,
                "generation": concept.generation,
                "proposal_uses": concept.proposal_uses,
                "useful_uses": concept.useful_uses,
            }
            for concept in self._concepts.values()
        ]
