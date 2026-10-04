from concept_recomposition.archive import ConceptArchive
from concept_recomposition.market_appendix import _summary
from concept_recomposition.search import CandidateRecord, SearchResult


def test_market_summary_counts_stable_depth() -> None:
    record = CandidateRecord(
        proposal=1,
        generation=0,
        expression="x1",
        expanded_key="raw:x1",
        expanded_size=1,
        expanded_depth=3,
        root_op="raw",
        discovery_score=0.08,
        confirmation_score=0.07,
        promotion_score=0.07,
        heldout_score=0.04,
        confirmation_gain=0.0,
        exact_target=False,
        target_key=None,
        concept_refs=(),
    )
    result = SearchResult(
        world="market",
        arm="reset",
        records=[record],
        archive=ConceptArchive(),
        first_target_proposal=None,
        first_success_proposal=None,
        required_targets=0,
        distinct_targets=(),
        useful_concepts=(),
        generation_ends=(1,),
        promotion_schedule=((),),
    )
    summary = _summary([result])
    assert summary["mean_stable_candidates"] == 1.0
    assert summary["mean_max_stable_depth"] == 3.0
