from concept_recomposition.search import SearchRunner
from concept_recomposition.worlds import build_world


def test_matched_budget() -> None:
    world = build_world("deep", seed=4)
    reify = SearchRunner(world, "reify", seed=4, proposal_budget=200).run()
    reset = SearchRunner(world, "reset", seed=4, proposal_budget=200).run()
    sham = SearchRunner(
        world,
        "sham",
        seed=4,
        proposal_budget=200,
        sham_schedule=reify.sham_schedule,
    ).run()
    assert len(reset.records) <= 200
    assert len(reify.records) <= 200
    assert len(sham.records) <= 200


def test_sham_matches_promotion_count_when_candidates_exist() -> None:
    world = build_world("deep", seed=2)
    reify = SearchRunner(world, "reify", seed=2, proposal_budget=500).run()
    sham = SearchRunner(
        world,
        "sham",
        seed=2,
        proposal_budget=500,
        sham_schedule=reify.sham_schedule,
    ).run()
    assert len(sham.archive) == sum(
        len(generation) for generation in reify.sham_schedule
    )


def test_decoy_can_expand_without_target() -> None:
    world = build_world("decoy", seed=3)
    result = SearchRunner(
        world, "reify", seed=3, proposal_budget=300
    ).run()
    assert not result.reached_target


def test_reification_changes_reachability() -> None:
    world = build_world("deep", seed=1)
    reset = SearchRunner(
        world, "reset", seed=1, proposal_budget=500
    ).run()
    reify = SearchRunner(
        world, "reify", seed=1, proposal_budget=500
    ).run()
    assert not reset.reached_target
    assert reify.reached_target
