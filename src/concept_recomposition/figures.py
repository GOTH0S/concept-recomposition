from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
FIGURES = Path("figures")
COLORS = {"reset": "0.25", "reify": "C0", "sham": "C1", "process": "C2"}


def _rows() -> list[dict[str, str]]:
    with (RESULTS / "summary.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def reachability() -> None:
    rows = _rows()
    worlds = {"deep", "reuse", "context"}
    budgets = sorted({int(row["budget"]) for row in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for arm in ("reset", "sham", "reify", "process"):
        values = []
        for budget in budgets:
            cell = [
                float(row["reach_rate"])
                for row in rows
                if row["arm"] == arm
                and row["world"] in worlds
                and int(row["budget"]) == budget
            ]
            values.append(float(np.mean(cell)))
        ax.plot(
            budgets,
            values,
            marker="o",
            label=arm.upper(),
            color=COLORS[arm],
        )
    ax.set_xlabel("proposal budget")
    ax.set_ylabel("target reach rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Reusing discovered concepts changes what search can reach")
    ax.legend(frameon=False)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "reachability.svg")
    plt.close(fig)



def lineage() -> None:
    payload = json.loads(
        (RESULTS / "lineage.json").read_text(encoding="utf-8")
    )
    if not payload:
        return
    target = payload["target_record"]
    concept_ids = target["concept_refs"]
    concepts = {row["concept_id"]: row for row in payload["concepts"]}
    labels = []
    for concept_id in concept_ids:
        row = concepts[concept_id]
        labels.append(f"{concept_id}\n{row['expression']}")
    labels.append(target["expression"])
    fig, ax = plt.subplots(figsize=(8.0, 2.5))
    ax.axis("off")
    xs = np.linspace(0.1, 0.9, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels)):
        ax.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "0.4",
            },
        )
        if index:
            ax.annotate(
                "",
                xy=(x - 0.08, 0.5),
                xytext=(xs[index - 1] + 0.08, 0.5),
                arrowprops={"arrowstyle": "->"},
            )
    ax.set_title(f"Example lineage: {payload['world']} world")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "lineage.svg")
    plt.close(fig)



def process_comparison() -> None:
    rows = _rows()
    worlds = {"deep", "reuse", "context"}
    budgets = sorted({int(row["budget"]) for row in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for arm in ("reify", "process"):
        values = []
        for budget in budgets:
            cells = [
                float(row["reach_rate"])
                for row in rows
                if row["arm"] == arm
                and row["world"] in worlds
                and int(row["budget"]) == budget
            ]
            values.append(float(np.mean(cells)))
        ax.plot(
            budgets,
            values,
            marker="o",
            label=arm.upper(),
            color=COLORS[arm],
        )
    ax.set_xlabel("proposal budget")
    ax.set_ylabel("target reach rate")
    ax.set_ylim(-0.01, 0.35)
    ax.set_title("Adaptive proposal bias changes target reach")
    ax.legend(frameon=False)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "process.svg")
    plt.close(fig)

def render_all() -> None:
    reachability()
    lineage()
    process_comparison()


if __name__ == "__main__":
    render_all()
