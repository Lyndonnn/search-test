#!/usr/bin/env python3
"""Recompute DAG-IG v3.1 totals from edge components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from grounded_pipeline_utils import load_jsonl, md_table, write_jsonl


SPLITS = ["train", "dev", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored_dir", default="results/dagig_rn03_10_counterfactual_dagig_v31")
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    return parser.parse_args()


def f(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None and key == "R_search_v3":
        value = row.get("R_search")
    if value is None and key == "R_answer_v31":
        value = row.get("R_answer")
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def cost(row: dict[str, Any]) -> float:
    if row.get("R_cost_v31") is not None:
        return max(0.0, float(row.get("R_cost_v31") or 0.0))
    return max(0.0, -float(row.get("R_cost") or 0.0))


def recompute(row: dict[str, Any]) -> dict[str, Any]:
    r_ground = f(row, "R_ground")
    r_observe = f(row, "R_observe")
    r_search = f(row, "R_search_v3")
    r_evidence = f(row, "R_evidence")
    r_answer = f(row, "R_answer_v31")
    r_cost = cost(row)
    total = 0.20 * r_ground + 0.15 * r_observe + 0.30 * r_search + 0.20 * r_evidence + 0.15 * r_answer - r_cost
    search_evidence_only = 0.50 * r_search + 0.50 * r_evidence - r_cost
    no_answer = 0.25 * r_ground + 0.20 * r_observe + 0.35 * r_search + 0.20 * r_evidence - r_cost
    answer_heavy = 0.15 * r_ground + 0.10 * r_observe + 0.25 * r_search + 0.20 * r_evidence + 0.30 * r_answer - r_cost
    return {
        **row,
        "DAGIG_v31_total": float(total),
        "DAGIG_v31_search_evidence_only": float(search_evidence_only),
        "DAGIG_v31_no_answer": float(no_answer),
        "DAGIG_v31_answer_heavy": float(answer_heavy),
        "DAGIG_v31_weights": {
            "R_ground": 0.20,
            "R_observe": 0.15,
            "R_search_v3": 0.30,
            "R_evidence": 0.20,
            "R_answer_v31": 0.15,
            "R_cost_v31": -1.0,
        },
    }


def summarize(rows_by_split: dict[str, list[dict[str, Any]]]) -> str:
    summary = []
    for split, rows in rows_by_split.items():
        n = len(rows)
        summary.append(
            {
                "split": split,
                "n": n,
                "DAGIG_v31_total": mean([f(r, "DAGIG_v31_total") for r in rows]) if n else 0.0,
                "search_evidence_only": mean([f(r, "DAGIG_v31_search_evidence_only") for r in rows]) if n else 0.0,
                "no_answer": mean([f(r, "DAGIG_v31_no_answer") for r in rows]) if n else 0.0,
                "answer_heavy": mean([f(r, "DAGIG_v31_answer_heavy") for r in rows]) if n else 0.0,
                "R_answer_v31": mean([f(r, "R_answer_v31") for r in rows]) if n else 0.0,
                "R_cost_v31": mean([cost(r) for r in rows]) if n else 0.0,
            }
        )
    return "# DAG-IG v3.1 Total Summary\n\n" + md_table(summary) + "\n"


def main() -> int:
    args = parse_args()
    scored_dir = Path(args.scored_dir)
    rows_by_split = {}
    for split in args.splits:
        path = scored_dir / f"scored_rollouts_{split}.jsonl"
        rows = [recompute(row) for row in load_jsonl(path)]
        write_jsonl(path, rows)
        rows_by_split[split] = rows
    (scored_dir / "dagig_total_v31_summary.md").write_text(summarize(rows_by_split), encoding="utf-8")
    print(json.dumps({split: len(rows) for split, rows in rows_by_split.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
