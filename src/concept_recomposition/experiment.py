from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .metrics import summarize
from .search import SearchResult, SearchRunner
from .worlds import WORLD_NAMES, build_world

ARMS = ("reset", "reify", "sham", "process")
BUDGETS = (25, 50, 100, 250, 500)


def run_cell(world_name: str, seed: int) -> dict[str, SearchResult]:
    world = build_world(world_name, seed=seed)
    budget = max(BUDGETS)
    reify = SearchRunner(world, "reify", seed=seed, proposal_budget=budget).run()
    return {
        "reset": SearchRunner(
            world, "reset", seed=seed, proposal_budget=budget
        ).run(),
        "reify": reify,
        "sham": SearchRunner(
            world,
            "sham",
            seed=seed,
            proposal_budget=budget,
            sham_schedule=reify.sham_schedule,
        ).run(),
        "process": SearchRunner(
            world, "process", seed=seed, proposal_budget=budget
        ).run(),
    }


def run_sweep(seeds: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    lineage: dict[str, object] = {}

    for world_name in WORLD_NAMES:
        buckets: dict[str, list[SearchResult]] = {arm: [] for arm in ARMS}
        for seed in range(seeds):
            cell = run_cell(world_name, seed)
            for arm, result in cell.items():
                buckets[arm].append(result)
                if (
                    not lineage
                    and world_name in {"deep", "reuse", "context"}
                    and arm == "reify"
                    and result.reached_target
                ):
                    target_record = next(
                        (record for record in result.records if record.exact_target),
                        None,
                    )
                    if target_record is not None and target_record.concept_refs:
                        observed_by = next(
                            budget
                            for budget in BUDGETS
                            if target_record.proposal <= budget
                        )
                        lineage = {
                            "world": world_name,
                            "arm": arm,
                            "observed_by_budget": observed_by,
                            "seed": seed,
                            "first_target_proposal": result.first_target_proposal,
                            "concepts": result.archive.rows(),
                            "target_record": target_record.__dict__,
                            "trajectory": [
                                record.__dict__ for record in result.records
                            ],
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the concept recomposition benchmark"
    )
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument(
        "--out", type=Path, default=Path("results/summary.csv")
    )
    parser.add_argument(
        "--lineage-out",
        type=Path,
        default=Path("results/lineage.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, lineage = run_sweep(args.seeds)
    write_csv(rows, args.out)
    args.lineage_out.parent.mkdir(parents=True, exist_ok=True)
    args.lineage_out.write_text(
        json.dumps(lineage, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
