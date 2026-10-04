from __future__ import annotations

import argparse
import csv
import hashlib
import io
import urllib.request
from pathlib import Path

import numpy as np

from .experiment import ARMS
from .search import SearchResult, SearchRunner
from .worlds import World

SOURCE_COMMIT = "3187e756957af549e54e470e8064dbe782d319b8"
SOURCE_URL = (
    "https://raw.githubusercontent.com/marcoreyess22/jump-risk-engine/"
    f"{SOURCE_COMMIT}/data/prices.csv"
)
SOURCE_SHA256 = "0f1c7534aed1afc5d431be99f5c0357243e7357905e60b74b40a8711689f71bd"


def _download(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = path.read_bytes()
    else:
        request = urllib.request.Request(
            SOURCE_URL,
            headers={"User-Agent": "concept-recomposition/0.1"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
        path.write_bytes(payload)
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError("price snapshot hash mismatch")
    return payload.decode("utf-8")


def build_market_world(cache: Path) -> tuple[World, int]:
    rows = list(csv.DictReader(io.StringIO(_download(cache))))
    keep = [row for row in rows if "2008-01-01" <= row["Date"] <= "2019-12-31"]
    tickers = ("SPY", "QQQ", "TLT", "GLD", "EEM")
    prices = np.array([[float(row[ticker]) for ticker in tickers] for row in keep])
    returns = np.full_like(prices, np.nan)
    returns[1:] = np.log(prices[1:] / prices[:-1])
    future_spy = np.full(returns.shape[0], np.nan)
    future_spy[:-1] = returns[1:, 0]
    data = {f"x{i + 1}": returns[:, i] for i in range(5)}
    split = next(i for i, row in enumerate(keep) if row["Date"] >= "2015-01-02")
    return World("market", data, (), (future_spy,)), split


def _summary(results: list[SearchResult]) -> dict[str, float]:
    max_depth = []
    stable_count = []
    unique_roots = []
    reused = []
    best = []
    for result in results:
        stable = [
            record
            for record in result.records
            if record.validation_score >= 0.05 and record.heldout_score >= 0.02
        ]
        max_depth.append(max((record.expanded_depth for record in stable), default=0))
        stable_count.append(len(stable))
        unique_roots.append(len({record.root_op for record in stable}))
        reused.append(sum(row["proposal_uses"] > 0 for row in result.archive.rows()))
        best.append(result.best_heldout)
    return {
        "mean_stable_candidates": float(np.mean(stable_count)),
        "mean_max_stable_depth": float(np.mean(max_depth)),
        "mean_stable_root_ops": float(np.mean(unique_roots)),
        "mean_reused_concepts": float(np.mean(reused)),
        "mean_best_heldout_corr": float(np.mean(best)),
    }


def run_market_appendix(
    cache: Path, seeds: int = 10, budget: int = 500
) -> list[dict[str, object]]:
    world, split = build_market_world(cache)
    buckets: dict[str, list[SearchResult]] = {arm: [] for arm in ARMS}
    for seed in range(seeds):
        reify = SearchRunner(
            world,
            "reify",
            seed=seed,
            proposal_budget=budget,
            promotion_threshold=0.05,
            split=split,
        ).run()
        runs = {
            "reset": SearchRunner(
                world,
                "reset",
                seed=seed,
                proposal_budget=budget,
                promotion_threshold=0.05,
                split=split,
            ).run(),
            "reify": reify,
            "sham": SearchRunner(
                world,
                "sham",
                seed=seed,
                proposal_budget=budget,
                promotion_threshold=0.05,
                split=split,
                sham_schedule=reify.sham_schedule,
            ).run(),
            "process": SearchRunner(
                world,
                "process",
                seed=seed,
                proposal_budget=budget,
                promotion_threshold=0.05,
                split=split,
            ).run(),
        }
        for arm, result in runs.items():
            buckets[arm].append(result)

    return [
        {"arm": arm, "seeds": seeds, "budget": budget, **_summary(buckets[arm])}
        for arm in ARMS
    ]


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stale-market appendix")
    parser.add_argument("--cache", type=Path, default=Path(".cache/prices.csv"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument(
        "--out", type=Path, default=Path("results/market_appendix.csv")
    )
    args = parser.parse_args()
    write_csv(
        run_market_appendix(args.cache, args.seeds, args.budget),
        args.out,
    )


if __name__ == "__main__":
    main()
