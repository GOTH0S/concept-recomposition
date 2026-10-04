import numpy as np

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


def test_nontrivial_targets_exceed_local_depth() -> None:
    for name in ("deep", "reuse", "context"):
        world = build_world(name)
        assert world.intermediates
        assert all(target.depth > 2 for target in world.targets)


def test_context_uses_one_observable_condition() -> None:
    world = build_world("context")
    assert len(world.conditions) == 1
    assert "where" in world.grammar_ops


def test_decoy_has_no_hidden_expression() -> None:
    assert not build_world("decoy").targets
