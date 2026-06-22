#!/usr/bin/env python3
"""Diagnose why counterfactual DAG-IG v1 did not give paper-level evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from grounded_pipeline_utils import md_table, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1_report", default="results/dagig_rn03_10_counterfactual_dagig/COUNTERFACTUAL_DAGIG_REPORT.md")
    parser.add_argument("--v1_comparison_csv", default="results/dagig_rn03_10_counterfactual_dagig/tables/final_counterfactual_dagig_comparison.csv")
    parser.add_argument("--v1_predictiveness_csv", default="results/dagig_rn03_10_counterfactual_dagig/score_predictiveness.csv")
    parser.add_argument("--out_dir", default="results/dagig_rn03_10_counterfactual_dagig_v2")
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key)
        if value in (None, "", "None"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def classify_predictiveness(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    strong: list[dict[str, Any]] = []
    weak: list[dict[str, Any]] = []
    for row in rows:
        spearman = as_float(row, "spearman")
        item = {
            "split": row.get("split"),
            "comparison": row.get("comparison"),
            "spearman": spearman,
            "auc": row.get("auc"),
            "top_bottom_gap": row.get("top_bottom_gap"),
        }
        if abs(spearman) >= 0.30:
            strong.append(item)
        elif row.get("comparison") in {
            "R_search vs R@5",
            "R_search vs MRR",
            "DAGIG_total vs R@5",
            "DAGIG_total vs evidence_support",
            "DAGIG_total vs answer F1",
        }:
            weak.append(item)
    return strong, weak


def method_rows(rows: list[dict[str, str]], method: str) -> dict[str, dict[str, str]]:
    return {str(row.get("split")): row for row in rows if row.get("method") == method}


def compare_generic_vs_counterfactual(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    generic = method_rows(rows, "generic_process_lowbudget_rl")
    cf = method_rows(rows, "counterfactual_dagig_rejection_sft")
    out = []
    for split in sorted(set(generic) & set(cf)):
        g = generic[split]
        c = cf[split]
        out.append(
            {
                "split": split,
                "metric": "R@5",
                "generic": as_float(g, "R@5"),
                "counterfactual": as_float(c, "R@5"),
                "delta_cf_minus_generic": as_float(c, "R@5") - as_float(g, "R@5"),
            }
        )
        out.append(
            {
                "split": split,
                "metric": "MRR",
                "generic": as_float(g, "MRR"),
                "counterfactual": as_float(c, "MRR"),
                "delta_cf_minus_generic": as_float(c, "MRR") - as_float(g, "MRR"),
            }
        )
        out.append(
            {
                "split": split,
                "metric": "evidence_support",
                "generic": as_float(g, "evidence_support"),
                "counterfactual": as_float(c, "evidence_support"),
                "delta_cf_minus_generic": as_float(c, "evidence_support") - as_float(g, "evidence_support"),
            }
        )
    return out


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictiveness = read_csv(args.v1_predictiveness_csv)
    comparison = read_csv(args.v1_comparison_csv)
    strong, weak = classify_predictiveness(predictiveness)
    generic_delta = compare_generic_vs_counterfactual(comparison)
    diagnosis = {
        "predictive_edges": strong,
        "weak_edges": weak,
        "generic_vs_counterfactual": generic_delta,
        "summary": {
            "best_predictive_signal": "R_ground predicts IoU/center-hit; R_evidence is moderately useful.",
            "weakest_signal": "R_search and DAGIG_total are weak predictors of downstream retrieval on dev.",
            "why_generic_matches_or_beats": (
                "generic-process directly rewards retrieval/evidence-style process metrics, while v1 DAG-IG "
                "uses weak lexical proxies for the search edge and noisy counterfactuals."
            ),
            "must_fix": [
                "audit and filter bad counterfactuals",
                "replace search-edge lexical proxy with retrieval-delta counterfactual credit",
                "add explicit evidence support verifier",
                "only train if v2 predictiveness passes thresholds",
            ],
        },
    }
    write_json(out_dir / "v1_failure_diagnosis.json", diagnosis)
    md = [
        "# DAG-IG v1 Failure Diagnosis",
        "",
        "## Predictive Edge Credits",
        "",
        md_table(strong) or "No strong edge credits found.",
        "",
        "## Weak Edge Credits",
        "",
        md_table(weak) or "No weak edge credits found.",
        "",
        "## Generic Process vs Counterfactual DAG-IG v1",
        "",
        md_table(generic_delta) or "Missing comparison rows.",
        "",
        "## Diagnosis",
        "",
        "- Strongest part: grounding edge; R_ground predicts IoU/center-hit.",
        "- Moderately useful part: evidence edge.",
        "- Weak part: search edge and total DAG-IG score, especially on dev.",
        "- Generic-process matches or beats v1 because it directly rewards retrieval and evidence support, while v1 search credit is a weak lexical proxy.",
        "- More training is not justified until counterfactual quality, search counterfactuals, and evidence verification are fixed.",
    ]
    (out_dir / "v1_failure_diagnosis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(diagnosis["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
