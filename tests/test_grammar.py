from concept_recomposition.expression import Expression, ValueType
from concept_recomposition.grammar import concept_pool, proposal_pool
from concept_recomposition.worlds import build_world


def test_proposal_pool_only_returns_series() -> None:
    pool = proposal_pool(("x1", "x2"))
    assert pool
    assert all(expression.value_type is ValueType.SERIES for expression in pool)


def test_concept_pool_requires_a_concept_reference() -> None:
    pool = concept_pool(("x1", "x2"), ("C000",))
    assert pool
    assert all(expression.concept_ids() for expression in pool)


def test_deep_target_becomes_local_after_reification() -> None:
    world = build_world("deep", seed=0)
    target = world.targets[0]
    base = proposal_pool(tuple(world.data))
    assert target.key not in {expression.expanded({}).key for expression in base}

    intermediate = Expression.unary("mean", Expression.raw("x2"), 5)
    expanded = {
        expression.expanded({"C000": intermediate}).key
        for expression in proposal_pool(tuple(world.data), ("C000",))
    }
    assert target.key in expanded
