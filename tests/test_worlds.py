import numpy as np

from concept_recomposition.expression import Expression
from concept_recomposition.grammar import concept_pool, proposal_pool
from concept_recomposition.worlds import WORLD_NAMES, build_world


def test_worlds_are_deterministic() -> None:
    for name in WORLD_NAMES:
        left = build_world(name, seed=7)
        right = build_world(name, seed=7)
        assert left.target_keys == right.target_keys
        assert all(
            np.allclose(a, b, equal_nan=True)
            for a, b in zip(left.outcomes, right.outcomes)
        )


def test_decoy_has_no_hidden_expression() -> None:
    world = build_world("decoy")
    assert not world.targets



def _keys(expressions):
    return {expression.key for expression in expressions}


def test_shallow_target_is_in_base_proposal_pool() -> None:
    world = build_world("shallow", seed=0)
    base = _keys(proposal_pool(tuple(world.data)))
    assert world.targets[0].key in base


def test_deep_target_requires_reified_intermediate() -> None:
    world = build_world("deep", seed=0)
    target = world.targets[0]
    inner = target.args[0]
    base = _keys(proposal_pool(tuple(world.data)))
    assert inner.key in base
    assert target.key not in base

    concepts = {"C000": inner}
    derived = concept_pool(tuple(world.data), tuple(concepts))
    expanded = {expression.expanded(concepts).key for expression in derived}
    assert target.key in expanded


def test_reuse_targets_share_one_reachable_intermediate() -> None:
    world = build_world("reuse", seed=0)
    inner = world.targets[0].args[0]
    assert inner == world.targets[1].args[0]

    base = _keys(proposal_pool(tuple(world.data)))
    assert inner.key in base
    assert all(target.key not in base for target in world.targets)

    concepts = {"C000": inner}
    derived = concept_pool(tuple(world.data), tuple(concepts))
    expanded = {expression.expanded(concepts).key for expression in derived}
    assert all(target.key in expanded for target in world.targets)


def test_context_target_requires_concept_and_uses_both_states() -> None:
    world = build_world("context", seed=0)
    target = world.targets[0]
    inner = target.args[1]
    base = _keys(proposal_pool(tuple(world.data)))
    assert inner.key in base
    assert target.key not in base

    positive_share = float(np.mean(world.data["x5"] > 0))
    assert 0.25 < positive_share < 0.75

    concepts = {"C000": inner}
    derived = concept_pool(tuple(world.data), tuple(concepts))
    expanded = {expression.expanded(concepts).key for expression in derived}
    assert target.key in expanded
