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


def test_reify_and_sham_spend_the_same_budget() -> None:
    world = build_world("deep", seed=4)
    reify = SearchRunner(
        world,
        "reify",
        seed=4,
        proposal_budget=300,
    ).run()
    sham = SearchRunner(
        world,
        "sham",
        seed=4,
        proposal_budget=300,
        promotion_schedule=reify.promotion_schedule,
    ).run()

    assert len(reify.records) == len(sham.records) == 300
    assert len(reify.promotion_schedule) == len(sham.promotion_schedule)


def test_sham_matches_reify_promotion_count() -> None:
    world = build_world("deep", seed=2)
    reify = SearchRunner(
        world,
        "reify",
        seed=2,
        proposal_budget=300,
    ).run()
    sham = SearchRunner(
        world,
        "sham",
        seed=2,
        proposal_budget=300,
        promotion_schedule=reify.promotion_schedule,
    ).run()

    assert len(sham.archive) == len(reify.archive)
    assert [len(batch) for batch in sham.promotion_schedule] == [
        len(batch) for batch in reify.promotion_schedule
    ]


def test_reification_can_expand_beyond_local_depth() -> None:
    result = SearchRunner(
        build_world("deep", seed=5),
        "reify",
        seed=5,
        proposal_budget=500,
    ).run()

    assert result.archive.ids
    assert max(record.expanded_depth for record in result.records) > 2


def test_decoy_has_no_exact_target() -> None:
    result = SearchRunner(
        build_world("decoy", seed=3),
        "reify",
        seed=3,
        proposal_budget=300,
    ).run()

    assert not result.distinct_targets
