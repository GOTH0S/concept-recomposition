import numpy as np

from concept_recomposition.expression import Expression
from concept_recomposition.operators import evaluate


def test_commutative_keys_match() -> None:
    x1 = Expression.raw("x1")
    x2 = Expression.raw("x2")
    assert Expression.binary("add", x1, x2).key == Expression.binary("add", x2, x1).key


def test_concept_expands_to_original_expression() -> None:
    x1 = Expression.raw("x1")
    expr = Expression.unary("mean", x1, 5)
    ref = Expression.concept("C000")
    assert ref.expanded({"C000": expr}) == expr


def test_concept_evaluation_matches_expression() -> None:
    data = {"x1": np.arange(30, dtype=float)}
    expr = Expression.unary("mean", Expression.raw("x1"), 5)
    direct = evaluate(expr, data)
    via_concept = evaluate(Expression.concept("C000"), data, {"C000": expr})
    np.testing.assert_allclose(direct, via_concept, equal_nan=True)


def test_subexpression_keys_include_intermediate_nodes() -> None:
    x1 = Expression.raw("x1")
    inner = Expression.unary("mean", x1, 5)
    outer = Expression.unary("mean", inner, 3)
    assert inner.key in outer.subexpression_keys()
    assert outer.key in outer.subexpression_keys()
