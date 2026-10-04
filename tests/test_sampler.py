import numpy as np

from concept_recomposition.expression import Expression, ValueType
from concept_recomposition.grammar import Grammar


def test_sampler_is_deterministic_and_typed() -> None:
    grammar = Grammar(("x1", "x2", "x3"), max_local_depth=2)
    left_rng = np.random.default_rng(17)
    right_rng = np.random.default_rng(17)

    left = [grammar.sample(left_rng).key for _ in range(100)]
    right = [grammar.sample(right_rng).key for _ in range(100)]

    assert left == right
    assert all(
        grammar.sample(np.random.default_rng(seed)).value_type is ValueType.SERIES
        for seed in range(20)
    )


def test_sampled_expression_respects_local_depth() -> None:
    grammar = Grammar(("x1", "x2"), max_local_depth=2)
    rng = np.random.default_rng(11)

    expressions = [grammar.sample(rng) for _ in range(500)]

    assert max(expression.depth for expression in expressions) <= 2


def test_concept_can_compress_deeper_expression() -> None:
    grammar = Grammar(("x1", "x2"), max_local_depth=2)
    concept_expression = Expression.unary(
        "mean",
        Expression.unary("diff", Expression.raw("x2"), 5),
        10,
    )
    rng = np.random.default_rng(3)

    sampled = [grammar.sample(rng, concept_ids=("C000",)) for _ in range(1000)]
    with_concept = next(
        expression
        for expression in sampled
        if "C000" in expression.concept_ids() and expression.depth >= 1
    )

    assert with_concept.depth <= 2
    assert with_concept.expanded({"C000": concept_expression}).depth > with_concept.depth
