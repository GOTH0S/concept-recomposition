import numpy as np

from concept_recomposition.worlds import (
    WORLD_NAMES,
    make_world,
)


def test_worlds_are_deterministic() -> None:
    for name in WORLD_NAMES:
        left = make_world(name, seed=12)
        right = make_world(name, seed=12)
        np.testing.assert_array_equal(
            left.y,
            right.y,
        )


def test_decoy_has_no_target() -> None:
    assert make_world("decoy", seed=0).target is None
