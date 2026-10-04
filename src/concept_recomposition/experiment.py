from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

from .metrics import summarize
from .search import SearchResult, SearchRunner
from .worlds import WORLD_NAMES, build_world

ARMS = ("reset", "reify", "sham", "process")
BUDGETS = (50, 100, 200, 350, 500)


def run_cell(world_name: str, seed: int) -> dict[str, SearchResult]:
    world = build_world(world_name, seed=seed)
    budget = max(BUDGETS)
    score_cache: dict[str, tuple[float, float]] = {}

    reify = SearchRunner(
        world,
        "reify",
        seed=seed,
        proposal_budget=budget,
        score_cache=score_cache,
    ).run()

    return {
        "reset": SearchRunner(
            world,
            "reset",
            seed=seed,
            proposal_budget=budget,
            score_cache=score_cache,
        ).run(),
        "reify": reify,
        "sham": SearchRunner(
            world,
            "sham",
            seed=seed,
            proposal_budget=budget,
            score_cache=score_cache,
            promotion_schedule=reify.promotion_schedule,
        ).run(),
        "process": SearchRunner(
            world,
            "process",
            seed=seed,
            proposal_budget=budget,
            score_cache=score_cache,
        ).run(),
    }


def run_sweep(seeds: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    if seeds < 1:
        raise ValueError("seeds must be positive")

    rows: list[dict[str, object]] = []
    lineage: dict[str, object] = {}

    with ProcessPoolExecutor(max_workers=min(4, seeds)) as executor:
        for world_name in WORLD_NAMES:
            buckets: dict[str, list[SearchResult]] = {
                arm: [] for arm in ARMS
            }
            cells = executor.map(
                run_cell,
                repeat(world_name),
                range(seeds),
            )
            for seed, cell in enumerate(cells):
                for arm, result in cell.items():
                    buckets[arm].append(result)
                    if (
                        not lineage
                        and world_name in {"deep", "reuse", "context"}
                        and arm == "reify"
                    ):
                        target = next(
                            (
                                record
                                for record in result.records
                                if record.exact_target
                                and record.concept_refs
                            ),
                            None,
                        )
                        if target is not None:
                            lineage = {
                                "world": world_name,
                                "seed": seed,
                                "first_target_proposal": result.first_target_proposal,
                                "concepts": result.archive.rows(),
                                "target_record": target.__dict__,
                            }

            for budget in BUDGETS:
                for arm in ARMS:
                    rows.append(
                        {
                            "world": world_name,
                            "budget": budget,
                            "arm": arm,
                            "seeds": seeds,
                            **summarize(buckets[arm], budget),
                        }
                    )

    return rows, lineage


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the concept recomposition benchmark"
    )
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/summary.csv"),
    )
    parser.add_argument(
        "--lineage-out",
        type=Path,
        default=Path("results/lineage.json"),
    )
    args = parser.parse_args()

    rows, lineage = run_sweep(args.seeds)
    write_csv(rows, args.out)
    args.lineage_out.parent.mkdir(parents=True, exist_ok=True)
    args.lineage_out.write_text(
        json.dumps(lineage, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
