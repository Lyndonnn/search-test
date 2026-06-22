#!/usr/bin/env python3
"""Validate whether counterfactual DAG-IG scores predict downstream success."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean
from typing import Any

from grounded_pipeline_utils import load_jsonl, md_table, write_csv


TESTS = [
    ("DAGIG_total", "retrieval_r5", "binary", "DAGIG_total vs R@5"),
    ("DAGIG_total", "evidence_support", "binary", "DAGIG_total vs evidence_support"),
    ("DAGIG_total", "answer_f1", "continuous", "DAGIG_total vs answer F1"),
    ("R_ground", "ground_iou", "continuous", "R_ground vs IoU"),
    ("R_ground", "ground_center_hit", "binary", "R_ground vs center-hit"),
    ("R_search", "retrieval_r5", "binary", "R_search vs R@5"),
    ("R_search", "retrieval_mrr", "continuous", "R_search vs MRR"),
    ("R_evidence", "evidence_support", "binary", "R_evidence vs evidence_support"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored_dir", default="results/dagig_rn03_10_counterfactual_dagig/scored")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--output_csv", default="results/dagig_rn03_10_counterfactual_dagig/score_predictiveness.csv")
    parser.add_argument("--output_md", default="results/dagig_rn03_10_counterfactual_dagig/score_predictiveness.md")
    return parser.parse_args()


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx <= 0 or deny <= 0:
        return None
    return num / (denx * deny)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rank(xs), rank(ys))


def auc(scores: list[float], labels: list[float]) -> float | None:
    pairs = [(s, int(bool(y))) for s, y in zip(scores, labels)]
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
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
    bottom = [v for _s, v in pairs[:q]]
    top = [v for _s, v in pairs[-q:]]
    return mean(top) - mean(bottom)


def metric_row(rows: list[dict[str, Any]], split: str, score_key: str, target_key: str, kind: str, label: str) -> dict[str, Any]:
    xs = []
    ys = []
    for row in rows:
        if row.get(score_key) is None or row.get(target_key) is None:
            continue
        xs.append(float(row.get(score_key) or 0.0))
        ys.append(float(row.get(target_key) or 0.0))
    return {
        "split": split,
        "comparison": label,
        "score_key": score_key,
        "target_key": target_key,
        "n": len(xs),
        "spearman": spearman(xs, ys),
        "top_bottom_gap": top_bottom_gap(xs, ys),
        "auc": auc(xs, ys) if kind == "binary" else None,
        "score_mean": mean(xs) if xs else None,
        "target_mean": mean(ys) if ys else None,
    }


def main() -> int:
    args = parse_args()
    out_rows = []
    for split in args.splits:
        path = Path(args.scored_dir) / f"{split}.jsonl"
        if not path.is_file():
            continue
        rows = load_jsonl(path)
        for score_key, target_key, kind, label in TESTS:
            out_rows.append(metric_row(rows, split, score_key, target_key, kind, label))
    write_csv(args.output_csv, out_rows)
    display = []
    for row in out_rows:
        display.append(
            {
                k: (f"{v:.4f}" if isinstance(v, float) else v)
                for k, v in row.items()
            }
        )
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(
        "# Counterfactual DAG-IG Score Predictiveness\n\n"
        + "Higher Spearman/top-bottom/AUC means the edge score is more predictive of downstream success.\n\n"
        + md_table(display)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
