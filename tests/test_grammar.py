from concept_recomposition.grammar import concept_pool, proposal_pool


def test_base_proposals_are_one_step_from_raws() -> None:
    pool = proposal_pool(("x1", "x2"))
    assert pool
    assert max(expression.depth for expression in pool) <= 2


def test_concept_pool_requires_a_concept_reference() -> None:
    pool = concept_pool(("x1", "x2"), ("C000",))
    assert pool
    assert all(expression.concept_ids() for expression in pool)
