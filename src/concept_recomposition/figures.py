from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path

RESULTS = Path("results")
FIGURES = Path("figures")
WORLDS = {"deep", "reuse", "context"}
X = {50: 102, 100: 231, 200: 360, 350: 489, 500: 618}


def _rows() -> list[dict[str, str]]:
    with (RESULTS / "summary.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_reach(rows: list[dict[str, str]], arm: str, budget: int) -> float:
    values = [
        float(row["reach_rate"])
        for row in rows
        if row["arm"] == arm
        and row["world"] in WORLDS
        and int(row["budget"]) == budget
    ]
    return sum(values) / len(values)


def _y(rate: float) -> int:
    return round(340 - rate * 1080)


def _points(rows: list[dict[str, str]], arm: str) -> str:
    return " ".join(
        f"{X[budget]},{_y(_mean_reach(rows, arm, budget))}"
        for budget in X
    )


def _frame(title: str, subtitle: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 420">
<rect width="720" height="420" fill="white"/>
<g font-family="Arial, sans-serif" fill="#111">
<text x="72" y="30" font-size="20" font-weight="600">{html.escape(title)}</text>
<text x="72" y="49" font-size="11" fill="#555">{html.escape(subtitle)}</text>
<line x1="72" y1="340" x2="690" y2="340" stroke="#777"/><line x1="72" y1="70" x2="72" y2="340" stroke="#777"/>
<g stroke="#eee"><line x1="72" y1="286" x2="690" y2="286"/><line x1="72" y1="232" x2="690" y2="232"/><line x1="72" y1="178" x2="690" y2="178"/><line x1="72" y1="124" x2="690" y2="124"/><line x1="72" y1="70" x2="690" y2="70"/></g>
<g font-size="11" fill="#555"><text x="48" y="344">0%</text><text x="42" y="290">5%</text><text x="36" y="236">10%</text><text x="36" y="182">15%</text><text x="36" y="128">20%</text><text x="36" y="74">25%</text>
<text x="92" y="360">50</text><text x="221" y="360">100</text><text x="350" y="360">200</text><text x="479" y="360">350</text><text x="608" y="360">500</text><text x="326" y="390">proposal budget</text></g>
{body}
</g></svg>
"""


def reachability() -> None:
    rows = _rows()
    body = f"""<polyline points="{_points(rows, "reify")}" fill="none" stroke="#1f5f99" stroke-width="2.5"/>
<polyline points="{_points(rows, "process")}" fill="none" stroke="#333" stroke-width="2.5"/>
<polyline points="{_points(rows, "sham")}" fill="none" stroke="#999" stroke-width="2.5" stroke-dasharray="6 4"/>
<line x1="492" y1="80" x2="516" y2="80" stroke="#1f5f99" stroke-width="2.5"/><text x="524" y="84" font-size="11">REIFY</text>
<line x1="492" y1="98" x2="516" y2="98" stroke="#333" stroke-width="2.5"/><text x="524" y="102" font-size="11">PROCESS</text>
<line x1="492" y1="116" x2="516" y2="116" stroke="#999" stroke-width="2.5" stroke-dasharray="6 4"/><text x="524" y="120" font-size="11">SHAM</text>
<text x="524" y="138" font-size="11">RESET: {_mean_reach(rows, "reset", 500):.0%}</text>"""
    FIGURES.mkdir(exist_ok=True)
    (FIGURES / "reachability.svg").write_text(
        _frame(
            "Recursive targets reached",
            "mean across deep, reuse and context problems",
            body,
        ),
        encoding="utf-8",
    )


def process_comparison() -> None:
    rows = _rows()
    reify = _mean_reach(rows, "reify", 500)
    process = _mean_reach(rows, "process", 500)
    body = f"""<polyline points="{_points(rows, "reify")}" fill="none" stroke="#1f5f99" stroke-width="2.7"/>
<polyline points="{_points(rows, "process")}" fill="none" stroke="#333" stroke-width="2.7"/>
<circle cx="618" cy="{_y(reify)}" r="4" fill="#1f5f99"/><circle cx="618" cy="{_y(process)}" r="4" fill="#333"/>
<line x1="505" y1="86" x2="530" y2="86" stroke="#1f5f99" stroke-width="2.7"/><text x="538" y="90" font-size="11">REIFY</text>
<line x1="505" y1="106" x2="530" y2="106" stroke="#333" stroke-width="2.7"/><text x="538" y="110" font-size="11">PROCESS</text>
<text x="458" y="160" font-size="11" fill="#555">500 proposals: {reify:.1%} vs {process:.1%}</text>"""
    FIGURES.mkdir(exist_ok=True)
    (FIGURES / "process.svg").write_text(
        _frame(
            "Learning the proposal bias does not help overall",
            "mean reach across deep, reuse and context problems",
            body,
        ),
        encoding="utf-8",
    )


def lineage() -> None:
    payload = json.loads(
        (RESULTS / "lineage.json").read_text(encoding="utf-8")
    )
    concept = payload["concepts"][0]
    target = payload["target_record"]
    raw_match = re.search(r"raw:([^,)]+)", target["expanded_key"])
    raw = raw_match.group(1) if raw_match else "raw"
    expanded = target["expanded_key"].replace("raw:", "")
    concept_id = concept["concept_id"]
    concept_expr = html.escape(concept["expression"])
    target_expr = html.escape(target["expression"])
    expanded = html.escape(expanded)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 190">
<rect width="760" height="190" fill="white"/>
<g font-family="Arial, sans-serif" fill="#111">
<text x="24" y="28" font-size="18" font-weight="600">One successful recomposition</text>
<text x="24" y="47" font-size="11" fill="#555">{html.escape(payload["world"])} problem, seed {payload["seed"]}</text>
<rect x="24" y="82" width="88" height="46" rx="4" fill="white" stroke="#777"/><text x="68" y="110" text-anchor="middle" font-size="13">{html.escape(raw)}</text>
<line x1="112" y1="105" x2="150" y2="105" stroke="#777"/><polygon points="150,105 142,101 142,109" fill="#777"/>
<rect x="156" y="70" width="172" height="70" rx="4" fill="white" stroke="#777"/><text x="242" y="99" text-anchor="middle" font-size="13">{concept_expr}</text><text x="242" y="120" text-anchor="middle" font-size="11" fill="#555">saved as {html.escape(concept_id)}</text>
<line x1="328" y1="105" x2="366" y2="105" stroke="#777"/><polygon points="366,105 358,101 358,109" fill="#777"/>
<rect x="372" y="70" width="172" height="70" rx="4" fill="white" stroke="#777"/><text x="458" y="99" text-anchor="middle" font-size="13">{target_expr}</text><text x="458" y="120" text-anchor="middle" font-size="11" fill="#555">proposal {target["proposal"]} · local size {target["local_size"]}</text>
<line x1="544" y1="105" x2="582" y2="105" stroke="#777"/><polygon points="582,105 574,101 574,109" fill="#777"/>
<rect x="588" y="70" width="150" height="70" rx="4" fill="#f5f5f5" stroke="#222"/><text x="663" y="98" text-anchor="middle" font-size="11">{expanded}</text><text x="663" y="120" text-anchor="middle" font-size="11" fill="#555">hidden target · size {target["expanded_size"]}</text>
</g></svg>
"""
    FIGURES.mkdir(exist_ok=True)
    (FIGURES / "lineage.svg").write_text(svg, encoding="utf-8")


def render_all() -> None:
    reachability()
    lineage()
    process_comparison()


if __name__ == "__main__":
    render_all()
