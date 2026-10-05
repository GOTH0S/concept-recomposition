import numpy as np

from concept_recomposition.evaluator import score_expression
from concept_recomposition.expression import Expression
from concept_recomposition.worlds import build_world


def test_evidence_roles_are_separate() -> None:
    world = build_world("shallow", seed=3)
    score = score_expression(
        world.targets[0],
        world,
        {},
        discovery_end=180,
        heldout_start=270,
    )
    assert score.discovery > 0.9
    assert score.confirmation > 0.9
    assert score.heldout > 0.9


def test_promotion_score_requires_both_development_slices() -> None:
    world = build_world("decoy", seed=7)
    expr = Expression.unary("mean", Expression.raw("x1"), 5)
    score = score_expression(
        expr,
        world,
        {},
        discovery_end=180,
        heldout_start=270,
    )
    assert score.promotion == min(score.discovery, score.confirmation)
    assert np.isfinite(score.promotion)


def test_affine_equivalence_accepts_sign_and_scale_changes() -> None:
    from concept_recomposition.evaluator import affine_equivalent

    x = np.linspace(-2.0, 2.0, 100)
    y = 3.5 - 2.0 * x
    assert affine_equivalent(x, y)


def test_affine_equivalence_rejects_different_shape() -> None:
    from concept_recomposition.evaluator import affine_equivalent

    x = np.linspace(-2.0, 2.0, 100)
    y = x**2
    assert not affine_equivalent(x, y)
