import numpy as np
import pytest

from concept_recomposition.expression import Expression, ValueType
from concept_recomposition.operators import evaluate


def test_commutative_keys_match() -> None:
    x1 = Expression.raw("x1")
    x2 = Expression.raw("x2")
    assert Expression.binary("add", x1, x2).key == Expression.binary("add", x2, x1).key


def test_where_requires_boolean_condition() -> None:
    with pytest.raises(TypeError):
        Expression.where(
            Expression.raw("x1"),
            Expression.raw("x2"),
            Expression.raw("x3"),
        )


def test_comparison_and_where_evaluate() -> None:
    data = {
        "x1": np.array([-1.0, 2.0, -3.0, 4.0]),
        "x2": np.zeros(4),
    }
    condition = Expression.compare(Expression.raw("x1"), Expression.raw("x2"))
    expression = Expression.where(
        condition,
        Expression.raw("x1"),
        Expression.unary("abs", Expression.raw("x1")),
    )

    assert condition.value_type is ValueType.BOOLEAN
    np.testing.assert_array_equal(evaluate(expression, data), np.array([1.0, 2.0, 3.0, 4.0]))


def test_concept_expands_and_evaluates() -> None:
    data = {"x1": np.arange(30, dtype=float)}
    expression = Expression.unary("mean", Expression.raw("x1"), 5)
    ref = Expression.concept("C000")

    assert ref.expanded({"C000": expression}) == expression
    np.testing.assert_allclose(
        evaluate(ref, data, {"C000": expression}),
        evaluate(expression, data),
        equal_nan=True,
    )


def test_rank_is_causal() -> None:
    base = np.array([1.0, 2.0, 3.0, 100.0])
    changed_future = np.array([1.0, 2.0, 3.0, -100.0])
    expression = Expression.unary("rank", Expression.raw("x1"))

    left = evaluate(expression, {"x1": base})
    right = evaluate(expression, {"x1": changed_future})

    np.testing.assert_array_equal(left[:3], right[:3])
