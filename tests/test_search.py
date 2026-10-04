from concept_recomposition.search import (
    SearchConfig,
    SearchRunner,
)
from concept_recomposition.worlds import make_world


def test_search_is_deterministic() -> None:
    config = SearchConfig(
        proposals=160,
        generation_size=80,
    )
    left = SearchRunner(
        make_world("deep", seed=4),
        "reify",
        config,
        4,
    ).run()
    right = SearchRunner(
        make_world("deep", seed=4),
        "reify",
        config,
        4,
    ).run()

    assert left.first_reach_attempt == right.first_reach_attempt
    assert [
        record.expression
        for record in left.records
    ] == [
        record.expression
        for record in right.records
    ]


def test_reset_promotes_nothing() -> None:
    result = SearchRunner(
        make_world("shallow", seed=0),
        "reset",
        SearchConfig(
            proposals=160,
            generation_size=80,
        ),
        0,
    ).run()
    assert result.concepts_promoted == 0


def test_reify_can_create_deeper_expression() -> None:
    result = SearchRunner(
        make_world("deep", seed=2),
        "reify",
        SearchConfig(),
        2,
    ).run()
    assert max(
        record.expanded_depth
        for record in result.records
    ) > 1
