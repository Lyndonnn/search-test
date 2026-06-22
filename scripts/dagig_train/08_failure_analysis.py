#!/usr/bin/env python3
"""Build failure taxonomy tables and qualitative examples from reward details."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details_jsonl", action="append", required=True)
    parser.add_argument("--out_dir", default="results/dagig_train/failure_analysis")
    parser.add_argument("--examples_per_failure", type=int, default=3)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def method_from_path(path: Path) -> str:
    name = path.name
    for suffix in ["_reward_details.jsonl", "_details.jsonl", ".jsonl"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    all_failures = sorted({str(row.get("failure_category", "")) for row in rows})
    fieldnames = ["method", "n"] + all_failures
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, items in sorted(grouped.items()):
            counts = Counter(str(item.get("failure_category", "")) for item in items)
            out = {"method": method, "n": len(items)}
            for failure in all_failures:
                out[failure] = counts[failure] / max(1, len(items))
            writer.writerow(out)
    print(f"wrote {path}")


def write_examples(path: Path, rows: list[dict[str, Any]], examples_per_failure: int) -> None:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row["method"]), str(row.get("failure_category", ""))
        if len(buckets[key]) < examples_per_failure:
            buckets[key].append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# DAG-IG Failure Examples\n\n")
        for (method, failure), items in sorted(buckets.items()):
            f.write(f"## {method} / {failure}\n\n")
            for item in items:
                f.write(f"### {item.get('sample_id')}\n\n")
                f.write(f"- answer_em: `{item.get('answer_em')}`\n")
                f.write(f"- query_anchor_hit: `{item.get('query_anchor_hit')}`\n")
                f.write(f"- retrieval_r5: `{item.get('retrieval_r5')}`\n")
                f.write(f"- spurious_success: `{item.get('spurious_success')}`\n")
                pred = item.get("prediction")
                if isinstance(pred, dict):
                    f.write(f"- observe: {pred.get('observe', '')[:300]}\n")
                    f.write(f"- search: `{pred.get('search', '')}`\n")
                    f.write(f"- answer: `{pred.get('answer', '')}`\n")
                f.write("\n")
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for raw_path in args.details_jsonl:
        path = Path(raw_path)
        method = method_from_path(path)
        for row in load_jsonl(path):
            row = dict(row)
            row["method"] = method
            rows.append(row)
    out_dir = Path(args.out_dir)
    write_summary(out_dir / "failure_summary.csv", rows)
    write_examples(out_dir / "qualitative_examples.md", rows, args.examples_per_failure)


if __name__ == "__main__":
    main()
