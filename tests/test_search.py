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


def test_reify_sham_and_process_spend_the_same_budget() -> None:
    world = build_world("deep", seed=4)
    reify = SearchRunner(world, "reify", seed=4, proposal_budget=300).run()
    runs = [
        reify,
        SearchRunner(
            world,
            "sham",
            seed=4,
            proposal_budget=300,
            promotion_schedule=reify.promotion_schedule,
        ).run(),
        SearchRunner(world, "process", seed=4, proposal_budget=300).run(),
    ]

    assert {len(result.records) for result in runs} == {300}


def test_sham_matches_reify_promotion_count() -> None:
    world = build_world("deep", seed=2)
    reify = SearchRunner(world, "reify", seed=2, proposal_budget=300).run()
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


def test_process_is_deterministic() -> None:
    world = build_world("reuse", seed=6)
    left = SearchRunner(world, "process", seed=6, proposal_budget=300).run()
    right = SearchRunner(world, "process", seed=6, proposal_budget=300).run()

    assert left.records == right.records
    assert left.promotion_schedule == right.promotion_schedule


def test_decoy_has_no_exact_target() -> None:
    result = SearchRunner(
        build_world("decoy", seed=3),
        "reify",
        seed=3,
        proposal_budget=300,
    ).run()

    assert not result.distinct_targets


def test_intermediate_diagnostics_do_not_change_search() -> None:
    world = build_world("deep", seed=7)
    result = SearchRunner(
        world,
        "reify",
        seed=7,
        proposal_budget=400,
    ).run()

    intermediate_keys = {expression.key for expression in world.intermediates}
    flagged = {
        record.expanded_key
        for record in result.records
        if record.exact_intermediate
    }
    assert flagged.issubset(intermediate_keys)
    assert set(result.promoted_intermediates).issubset(intermediate_keys)
