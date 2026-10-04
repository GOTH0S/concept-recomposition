import numpy as np

from concept_recomposition.expression import Expression
from concept_recomposition.grammar import Grammar
from concept_recomposition.proposer import MotifModel, operator_transitions


def test_operator_transitions_capture_tree_structure() -> None:
    expression = Expression.unary(
        "mean",
        Expression.unary("diff", Expression.raw("x2"), 5),
        10,
    )

    assert operator_transitions(expression) == (
        ("__root__", "mean"),
        ("mean", "diff"),
    )


def test_motif_bias_changes_proposal_distribution() -> None:
    model = MotifModel()
    template = Expression.unary("mean", Expression.raw("x1"), 5)
    for _ in range(20):
        model.observe(template, 1.0)

    grammar = Grammar(("x1", "x2", "x3"), max_local_depth=2)
    rng = np.random.default_rng(8)
    roots = [
        grammar.sample(rng, transition_weights=model.weights).op
        for _ in range(400)
    ]

    assert roots.count("mean") > 100
