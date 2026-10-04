# concept-recomposition

Can a search system make new ideas easier to find by turning useful intermediate expressions into reusable building blocks?

This repo tests that directly. It searches a small mathematical language under a fixed proposal budget. Some search runs can name a useful expression and use it as a new primitive later; others cannot.

```text
raw variables -> candidate expressions -> useful intermediate -> new primitive -> later expressions
```

Four search rules compete:

- **RESET** — always searches from the original vocabulary.
- **REIFY** — useful derived expressions can become reusable primitives.
- **SHAM** — gets the same number of extra primitives as REIFY, but from low-scoring expressions.
- **PROCESS** — REIFY plus a simple bias toward operators that appeared in successful promoted expressions.

The benchmark has five synthetic worlds: one shallow target, one deep target, one shared intermediate reused by two targets, one conditional target, and one decoy world with no hidden structure to find.

The main quantity is simple: **did the search reach the hidden expression before it ran out of proposals?**

Results are generated from code and written to `results/`.

## Run

```bash
python -m pip install -e ".[dev]"
python -m concept_recomposition.experiment --seeds 30
```

Python 3.11+. Apache-2.0.
