#!/usr/bin/env python3
"""Validate DAG-IG scores against answer-equivalence verifier v3 labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from grounded_pipeline_utils import load_jsonl, md_table, write_csv


SPLITS = ["train", "dev", "test"]
TESTS = [
    ("DAGIG_v31_total", "supported_answer_soft", "continuous", "DAGIG_v31_total -> supported_answer_soft"),
    ("DAGIG_v31_total", "supported_answer_hard", "binary", "DAGIG_v31_total -> supported_answer_hard_v3"),
    ("DAGIG_v31_search_evidence_only", "evidence_support", "binary", "search_evidence_only -> evidence_support"),
    ("R_answer_v31", "answer_correct_score", "continuous", "R_answer_v31 -> answer_correct_score"),
    ("R_answer_v31", "supported_answer_soft", "continuous", "R_answer_v31 -> supported_answer_soft"),
    ("answer_f1_v2", "answer_correct_score", "continuous", "answer F1 -> answer_correct_score"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored_dir", default="results/dagig_rn03_10_counterfactual_dagig_v31")
    parser.add_argument("--answer_dir", default="results/dagig_rn03_10_answer_equivalence_v3")
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    parser.add_argument("--output_csv", default="results/dagig_rn03_10_answer_equivalence_v3/dagig_answer_equivalence_predictiveness.csv")
    parser.add_argument("--output_md", default="results/dagig_rn03_10_answer_equivalence_v3/dagig_answer_equivalence_predictiveness.md")
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return float(value)
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = math.sqrt(sum(x * x for x in dx)) * math.sqrt(sum(y * y for y in dy))
    if den <= 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / den


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rank(xs), rank(ys))


def auc(scores: list[float], labels: list[float]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if bool(y)]
    neg = [s for s, y in zip(scores, labels) if not bool(y)]
    if not pos or not neg:
        return None
    wins = 0.0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def top_bottom_gap(scores: list[float], values: list[float]) -> float | None:
    if len(scores) < 4:
        return None
    pairs = sorted(zip(scores, values), key=lambda item: item[0])
    q = max(1, len(pairs) // 4)
    return mean([v for _s, v in pairs[-q:]]) - mean([v for _s, v in pairs[:q]])


def load_joined(scored_path: Path, answer_path: Path) -> list[dict[str, Any]]:
    answer_rows = {
        (str(row.get("sample_id")), int(row.get("rollout_index") or 0)): row
        for row in load_jsonl(answer_path)
    }
    joined = []
    for row in load_jsonl(scored_path):
        key = (str(row.get("sample_id")), int(row.get("rollout_index") or 0))
        answer = answer_rows.get(key)
        if answer is None:
            continue
        joined.append({**row, **answer})
    return joined


def metric_values(rows: list[dict[str, Any]], score_key: str, target_key: str) -> tuple[list[float], list[float]]:
    xs = []
    ys = []
    for row in rows:
        x = as_float(row.get(score_key))
        y = as_float(row.get(target_key))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def metric_row(rows: list[dict[str, Any]], split: str, score_key: str, target_key: str, kind: str, label: str) -> dict[str, Any]:
    xs, ys = metric_values(rows, score_key, target_key)
    return {
        "split": split,
        "comparison": label,
        "score_key": score_key,
        "target_key": target_key,
        "n": len(xs),
        "spearman": spearman(xs, ys),
        "pearson": pearson(xs, ys),
        "auc": auc(xs, ys) if kind == "binary" else None,
        "top_bottom_gap": top_bottom_gap(xs, ys),
        "score_mean": mean(xs) if xs else None,
        "target_mean": mean(ys) if ys else None,
    }


def type_rows(rows_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            grouped.setdefault((split, str(row.get("answer_type") or "unknown")), []).append(row)
    out = []
    for (split, answer_type), rows in sorted(grouped.items()):
        n = len(rows)
        out.append(
            {
                "split": split,
                "answer_type": answer_type,
                "n": n,
                "supported_answer_hard_v3_positive_rate": mean([float(row.get("supported_answer_hard") is True) for row in rows]) if n else 0.0,
                "supported_answer_soft_mean": mean([float(row.get("supported_answer_soft") or 0.0) for row in rows]) if n else 0.0,
                "semantic_equivalent_rate": mean([float(row.get("semantic_equivalent") is True) for row in rows]) if n else 0.0,
                "answer_correct_score_mean": mean([float(row.get("answer_correct_score") or 0.0) for row in rows]) if n else 0.0,
            }
        )
    return out


def fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def main() -> int:
    args = parse_args()
    rows_by_split = {}
    metric_rows = []
    for split in args.splits:
        rows = load_joined(
            Path(args.scored_dir) / f"scored_rollouts_{split}.jsonl",
            Path(args.answer_dir) / f"answer_equivalence_{split}.jsonl",
        )
        rows_by_split[split] = rows
        for score_key, target_key, kind, label in TESTS:
            metric_rows.append(metric_row(rows, split, score_key, target_key, kind, label))
    write_csv(args.output_csv, metric_rows)
    display = [{k: fmt(v) if isinstance(v, float) else v for k, v in row.items()} for row in metric_rows]
    type_display = [{k: fmt(v) if isinstance(v, float) else v for k, v in row.items()} for row in type_rows(rows_by_split)]
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(
        "# DAG-IG With Answer Equivalence v3 Predictiveness\n\n"
        + md_table(display)
        + "\n\n## Positive Rate By Answer Type\n\n"
        + md_table(type_display)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": {split: len(rows) for split, rows in rows_by_split.items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
