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


def test_decoy_has_no_hidden_expression() -> None:
    world = build_world("decoy")
    assert not world.targets
