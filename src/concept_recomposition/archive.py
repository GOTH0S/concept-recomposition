from __future__ import annotations

from dataclasses import dataclass

from .expression import Expression


@dataclass
class Concept:
    concept_id: str
    expression: Expression
    validation_score: float
    generation: int
    proposal_uses: int = 0
    useful_uses: int = 0


class ConceptArchive:
    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {}
        self._by_expression: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._concepts)

    @property
    def expressions(self) -> dict[str, Expression]:
        return {key: concept.expression for key, concept in self._concepts.items()}

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._concepts)

    def add(self, expression: Expression, score: float, generation: int) -> Concept | None:
        key = expression.key
        if key in self._by_expression:
            return None
        concept_id = f"C{len(self._concepts):03d}"
        concept = Concept(concept_id, expression, score, generation)
        self._concepts[concept_id] = concept
        self._by_expression[key] = concept_id
        return concept

    def note_use(self, concept_ids: tuple[str, ...], useful: bool) -> None:
        for concept_id in set(concept_ids):
            concept = self._concepts[concept_id]
            concept.proposal_uses += 1
            if useful:
                concept.useful_uses += 1

    def lineage(self, concept_ids: tuple[str, ...]) -> set[str]:
        useful = set(concept_ids)
        pending = list(concept_ids)
        while pending:
            concept_id = pending.pop()
            for parent in self._concepts[concept_id].expression.concept_ids():
                if parent not in useful:
                    useful.add(parent)
                    pending.append(parent)
        return useful

    def false_expansion_rate(self, useful_concepts: tuple[str, ...] = ()) -> float:
        if not self._concepts:
            return 0.0
        useful = self.lineage(useful_concepts) if useful_concepts else set()
        return 1.0 - len(useful) / len(self._concepts)

    def rows(self) -> list[dict[str, object]]:
        return [
            {
                "concept_id": concept.concept_id,
                "expression": str(concept.expression),
                "validation_score": concept.validation_score,
                "generation": concept.generation,
                "proposal_uses": concept.proposal_uses,
                "useful_uses": concept.useful_uses,
            }
            for concept in self._concepts.values()
        ]
