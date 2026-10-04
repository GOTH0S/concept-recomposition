from concept_recomposition.grammar import concept_pool, proposal_pool


def test_base_proposals_are_one_step_from_raws() -> None:
    pool = proposal_pool(("x1", "x2"))
    assert pool
    assert max(expression.depth for expression in pool) <= 2


def test_concept_pool_requires_a_concept_reference() -> None:
    pool = concept_pool(("x1", "x2"), ("C000",))
    assert pool
    assert all(expression.concept_ids() for expression in pool)


def test_deep_target_becomes_local_after_reification() -> None:
    from concept_recomposition.expression import Expression
    from concept_recomposition.worlds import build_world

    world = build_world("deep", seed=0)
    target = world.targets[0]

    base = proposal_pool(tuple(world.data))
    assert target.key not in {expression.key for expression in base}

    intermediate = Expression.unary("mean", Expression.raw("x2"), 5)
    expanded = {
        expression.expanded({"C000": intermediate}).key
        for expression in proposal_pool(tuple(world.data), ("C000",))
    }
    assert target.key in expanded
