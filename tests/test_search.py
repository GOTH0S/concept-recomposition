from concept_recomposition.search import SearchRunner
from concept_recomposition.worlds import build_world


def test_reset_spends_exact_budget() -> None:
    result = SearchRunner(
        build_world("deep", seed=4),
        "reset",
        seed=4,
        proposal_budget=215,
    ).run()

    assert len(result.records) == 215
    assert result.generation_ends == (50, 100, 150, 200, 215)


def test_reset_is_deterministic() -> None:
    world = build_world("deep", seed=7)
    left = SearchRunner(world, "reset", seed=9, proposal_budget=100).run()
    right = SearchRunner(world, "reset", seed=9, proposal_budget=100).run()

    assert left.records == right.records


def test_deep_target_is_outside_reset_local_depth() -> None:
    world = build_world("deep", seed=1)
    assert min(target.depth for target in world.targets) > 2

    result = SearchRunner(
        world,
        "reset",
        seed=1,
        proposal_budget=1000,
        local_depth=2,
    ).run()

    assert result.first_target_proposal is None
    assert max(record.expanded_depth for record in result.records) <= 2
    assert not result.archive.ids


def test_decoy_has_no_exact_target() -> None:
    result = SearchRunner(
        build_world("decoy", seed=3),
        "reset",
        seed=3,
        proposal_budget=300,
    ).run()

    assert not result.distinct_targets
