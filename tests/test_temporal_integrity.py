import numpy as np

from concept_recomposition.expression import Expression
from concept_recomposition.grammar import concept_pool, proposal_pool
from concept_recomposition.operators import evaluate


def _dataset(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        f"x{i}": rng.normal(size=160)
        for i in range(1, 6)
    }


def _mutate_future(
    data: dict[str, np.ndarray],
    cutoff: int,
) -> dict[str, np.ndarray]:
    changed = {name: values.copy() for name, values in data.items()}
    for index, values in enumerate(changed.values(), start=1):
        values[cutoff:] = index * 1000.0 + np.arange(values.size - cutoff)
    return changed


def _assert_prefix_invariant(
    expressions,
    base: dict[str, np.ndarray],
    changed: dict[str, np.ndarray],
    cutoff: int,
    concepts: dict[str, Expression] | None = None,
) -> None:
    for expression in expressions:
        left = evaluate(expression, base, concepts, {})
        right = evaluate(expression, changed, concepts, {})
        np.testing.assert_allclose(
            left[:cutoff],
            right[:cutoff],
            equal_nan=True,
            err_msg=str(expression),
        )


def test_primitive_grammar_does_not_read_future_observations() -> None:
    cutoff = 100
    base = _dataset()
    changed = _mutate_future(base, cutoff)
    names = tuple(base)

    _assert_prefix_invariant(
        proposal_pool(names),
        base,
        changed,
        cutoff,
    )


def test_reified_compositions_preserve_prefix_invariance() -> None:
    cutoff = 100
    base = _dataset()
    changed = _mutate_future(base, cutoff)
    names = tuple(base)
    concept = Expression.unary("mean", Expression.raw("x1"), 5)
    concepts = {"C000": concept}

    _assert_prefix_invariant(
        concept_pool(names, tuple(concepts)),
        base,
        changed,
        cutoff,
        concepts,
    )
