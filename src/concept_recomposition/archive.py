from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .expression import Expression
from .operators import evaluate

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class Concept:
    name: str
    expression: Expression
    generation: int
    validation_score: float
    reuse_count: int = 0


@dataclass(slots=True)
class ConceptArchive:
    concepts: dict[str, Concept] = field(default_factory=dict)

    def add(
        self,
        expression: Expression,
        generation: int,
        validation_score: float,
    ) -> Concept:
        name = f"C{len(self.concepts):03d}"
        concept = Concept(name, expression, generation, validation_score)
        self.concepts[name] = concept
        return concept

    def terminals(self, limit: int | None = None) -> tuple[Expression, ...]:
        concepts = list(self.concepts.values())
        concepts.sort(key=lambda concept: concept.validation_score, reverse=True)
        if limit is not None:
            concepts = concepts[:limit]
        return tuple(Expression.concept(concept.name) for concept in concepts)

    def values(
        self,
        variables: dict[str, FloatArray],
        cache: dict[Expression, FloatArray] | None = None,
    ) -> dict[str, FloatArray]:
        resolved: dict[str, FloatArray] = {}
        for name, concept in self.concepts.items():
            resolved[name] = evaluate(
                concept.expression,
                variables,
                resolved,
                cache,
            )
        return resolved

    def expand(self, expr: Expression) -> Expression:
        if expr.kind == "var":
            return expr
        if expr.kind == "concept":
            return self.expand(self.concepts[expr.name].expression)
        expanded = tuple(self.expand(arg) for arg in expr.args)
        if expr.kind == "unary":
            return Expression.unary(expr.name, expanded[0], expr.param)
        if expr.kind == "binary":
            return Expression.binary(expr.name, expanded[0], expanded[1])
        return Expression.where(expanded[0], expanded[1], expanded[2])

    def expanded_depth(self, expr: Expression) -> int:
        return self.expand(expr).local_depth

    def mark_reuse(self, expr: Expression) -> None:
        for name in self.concept_names(expr):
            self.concepts[name].reuse_count += 1

    def concept_names(self, expr: Expression) -> set[str]:
        if expr.kind == "concept":
            return {expr.name}
        found: set[str] = set()
        for arg in expr.args:
            found.update(self.concept_names(arg))
        return found

    def lineage_concepts(self, expr: Expression) -> set[str]:
        direct = self.concept_names(expr)
        found = set(direct)
        for name in direct:
            found.update(self.lineage_concepts(self.concepts[name].expression))
        return found
