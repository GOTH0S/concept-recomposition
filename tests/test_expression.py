import numpy as np

from concept_recomposition.archive import ConceptArchive
from concept_recomposition.expression import Expression
from concept_recomposition.operators import evaluate


def test_commutative_expression_is_canonical() -> None:
    x1 = Expression.var("x1")
    x2 = Expression.var("x2")
    assert Expression.binary("add", x1, x2) == Expression.binary("add", x2, x1)


def test_concept_expands_depth() -> None:
    archive = ConceptArchive()
    base = Expression.unary("mean", Expression.var("x1"), 5)
    concept = archive.add(base, generation=0, validation_score=1.0)
    child = Expression.unary("diff", Expression.concept(concept.name), 3)
    assert child.local_depth == 1
    assert archive.expanded_depth(child) == 2


def test_concept_evaluation_matches_definition() -> None:
    values = {"x1": np.arange(20, dtype=float)}
    archive = ConceptArchive()
    expr = Expression.unary("mean", Expression.var("x1"), 3)
    concept = archive.add(expr, 0, 1.0)
    resolved = archive.values(values)
    direct = evaluate(expr, values, {})
    np.testing.assert_allclose(
        resolved[concept.name],
        direct,
        equal_nan=True,
    )
