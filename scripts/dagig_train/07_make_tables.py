#!/usr/bin/env python3
"""Make compact CSV/Markdown result tables from DAG-IG experiment outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="results/dagig_train")
    parser.add_argument("--out_dir", default="results/dagig_train/tables")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    priority = ["method", "variant", "n"]
    fieldnames = priority + [key for key in fieldnames if key not in priority]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("_No rows._\n", encoding="utf-8")
        return
    fieldnames = ["method", "variant", "n"]
    metric_order = [
        "answer_em",
        "answer_f1",
        "query_anchor_hit",
        "query_specificity",
        "retrieval_r1",
        "retrieval_r5",
        "retrieval_mrr",
        "unsupported_answer",
        "spurious_success",
        "search_call",
        "reward_dagig",
    ]
    for key in metric_order:
        if any(key in row for row in rows):
            fieldnames.append(key)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(key, "")) for key in fieldnames) + " |\n")
    print(f"wrote {path}")


def collect(results_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(results_dir.glob("*_eval.csv")):
        for row in read_csv(path):
            row = dict(row)
            row["method"] = path.name[: -len("_eval.csv")]
            rows.append(row)
    for path in sorted(results_dir.glob("*_reward_summary.csv")):
        for row in read_csv(path):
            row = dict(row)
            row["method"] = path.name[: -len("_reward_summary.csv")]
            rows.append(row)
    for path in sorted(results_dir.glob("*_query_retrieval.csv")):
        for row in read_csv(path):
            row = dict(row)
            row["method"] = path.name[: -len("_query_retrieval.csv")]
            if "retrieval_top1" in row:
                row["retrieval_r1"] = row["retrieval_top1"]
            if "retrieval_top5" in row:
                row["retrieval_r5"] = row["retrieval_top5"]
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    rows = collect(results_dir)
    write_csv(out_dir / "all_results.csv", rows)
    write_markdown(out_dir / "all_results.md", rows)


if __name__ == "__main__":
    main()
