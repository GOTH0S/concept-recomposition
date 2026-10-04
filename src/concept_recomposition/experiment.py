from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .search import Arm, SearchConfig, SearchRunner
from .worlds import WORLD_NAMES, make_world

ARMS: tuple[Arm, ...] = (
    "reset",
    "reify",
    "process",
    "sham",
)
BUDGETS = (80, 160, 320, 640)


def run_sweep(
    *,
    seeds: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for world_name in WORLD_NAMES:
        for arm in ARMS:
            results = []
            for seed in range(seeds):
                world = make_world(world_name, seed=seed)
                results.append(
                    SearchRunner(
                        world,
                        arm,
                        SearchConfig(),
                        seed,
                    ).run()
                )

            for budget in BUDGETS:
                reached = [
                    result.first_reach_attempt is not None
                    and result.first_reach_attempt <= budget
                    for result in results
                ]
                first = [
                    result.first_reach_attempt
                    for result in results
                    if result.first_reach_attempt is not None
                    and result.first_reach_attempt <= budget
                ]
                rows.append(
                    {
                        "world": world_name,
                        "arm": arm,
                        "budget": budget,
                        "seeds": seeds,
                        "reach_rate": float(np.mean(reached)),
                        "mean_attempts_to_reach": (
                            float(np.mean(first))
                            if first
                            else ""
                        ),
                        "mean_concepts_promoted": float(
                            np.mean(
                                [
                                    result.concepts_promoted
                                    for result in results
                                ]
                            )
                        ),
                        "mean_concepts_reused": float(
                            np.mean(
                                [
                                    result.concepts_reused
                                    for result in results
                                ]
                            )
                        ),
                        "mean_false_expansion_rate": float(
                            np.mean(
                                [
                                    result.false_expansion_rate
                                    for result in results
                                ]
                            )
                        ),
                        "mean_max_useful_depth": float(
                            np.mean(
                                [
                                    result.max_useful_depth
                                    for result in results
                                ]
                            )
                        ),
                    }
                )
    return rows


def write_csv(
    rows: list[dict[str, float | int | str]],
    out: Path,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run concept recomposition benchmark"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/sweep.csv"),
    )
    args = parser.parse_args()
    write_csv(
        run_sweep(seeds=args.seeds),
        args.out,
    )


if __name__ == "__main__":
    main()
