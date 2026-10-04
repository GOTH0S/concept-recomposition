from concept_recomposition.search import SearchRunner
from concept_recomposition.worlds import build_world


def test_every_arm_spends_the_same_budget() -> None:
    world = build_world("deep", seed=4)
    reify = SearchRunner(
        world,
        "reify",
        seed=4,
        proposal_budget=200,
    ).run()
    runs = [
        SearchRunner(
            world,
            "reset",
            seed=4,
            proposal_budget=200,
        ).run(),
        reify,
        SearchRunner(
            world,
            "sham",
            seed=4,
            proposal_budget=200,
            promotion_schedule=reify.promotion_schedule,
        ).run(),
        SearchRunner(
            world,
            "process",
            seed=4,
            proposal_budget=200,
        ).run(),
    ]
    assert {len(result.records) for result in runs} == {200}


def test_sham_matches_reify_promotion_count() -> None:
    world = build_world("deep", seed=2)
    reify = SearchRunner(
        world,
        "reify",
        seed=2,
        proposal_budget=500,
    ).run()
    sham = SearchRunner(
        world,
        "sham",
        seed=2,
        proposal_budget=500,
        promotion_schedule=reify.promotion_schedule,
    ).run()
    assert len(sham.archive) == len(reify.archive)


def test_reset_never_uses_concepts() -> None:
    result = SearchRunner(
        build_world("deep", seed=1),
        "reset",
        seed=1,
        proposal_budget=200,
    ).run()
    assert not result.archive.ids
    assert all(not record.concept_refs for record in result.records)


def test_high_confirmation_threshold_blocks_decoy_growth() -> None:
    result = SearchRunner(
        build_world("decoy", seed=3),
        "reify",
        seed=3,
        proposal_budget=300,
        promotion_threshold=0.99,
    ).run()
    assert not result.archive.ids


def test_reified_proposals_can_exceed_local_depth() -> None:
    result = SearchRunner(
        build_world("deep", seed=1),
        "reify",
        seed=1,
        proposal_budget=500,
    ).run()
    assert max(record.expanded_depth for record in result.records) > 2


def test_reuse_world_requires_both_downstream_targets() -> None:
    world = build_world("reuse", seed=0)
    assert len(world.targets) == 2
    assert world.required_targets == 2


def test_reify_promotes_at_most_one_concept_per_generation() -> None:
    result = SearchRunner(
        build_world("deep", seed=5),
        "reify",
        seed=5,
        proposal_budget=500,
    ).run()
    assert len(result.archive) <= 10
