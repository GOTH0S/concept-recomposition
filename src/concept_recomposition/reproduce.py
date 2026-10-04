from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .experiment import BUDGETS, run_sweep, write_csv
from .figures import render_all
from .market_appendix import run_market_appendix
from .market_appendix import write_csv as write_market


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the headline results")
    parser.add_argument("--market", action="store_true")
    args = parser.parse_args()

    rows, lineage = run_sweep(50)
    write_csv(rows, Path("results/summary.csv"))
    headline_fields = (
        "world",
        "budget",
        "arm",
        "seeds",
        "reach_rate",
        "intermediate_reach_rate",
        "intermediate_promotion_rate",
        "mean_first_target",
        "mean_concepts",
        "mean_false_expansion",
        "mean_reused_concepts",
        "mean_max_target_depth",
        "mean_selected_heldout",
        "mean_selection_regret",
    )
    headline = [
        {field: row[field] for field in headline_fields}
        for row in rows
        if (
            row["budget"] == max(BUDGETS)
            or (
                row["world"] in {"deep", "reuse", "context"}
                and row["arm"] in {"reify", "process"}
            )
        )
    ]
    with Path("results/headline.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=headline_fields,
        )
        writer.writeheader()
        writer.writerows(headline)

    Path("results/lineage.json").write_text(
        json.dumps(lineage, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.market:
        write_market(
            run_market_appendix(Path(".cache/prices.csv")),
            Path("results/market_appendix.csv"),
        )
    render_all()


if __name__ == "__main__":
    main()
