from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run_sweep, write_csv
from .figures import render_all
from .market_appendix import run_market_appendix
from .market_appendix import write_csv as write_market


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the headline results")
    parser.add_argument("--market", action="store_true")
    args = parser.parse_args()

    rows, lineage = run_sweep(30)
    write_csv(rows, Path("results/summary.csv"))
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
