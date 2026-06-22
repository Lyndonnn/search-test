#!/usr/bin/env python3
"""Build a fixed offline evidence corpus for Pix2Fact-DAGIG retrieval/RL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_PACKAGE_NAME = "pix2fact_dagig_1k_gpt54_teacher_clean_package"
DEFAULT_MAIN_FILE = "data/pix2fact_dagig_train_AB_clean_split.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package_dir", default=f"data/{DEFAULT_PACKAGE_NAME}")
    parser.add_argument("--main_file", default=DEFAULT_MAIN_FILE)
    parser.add_argument("--out_dir", default="data/dagig_retrieval")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path} n={len(rows)}")


def teacher(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("gpt54_teacher")
    return value if isinstance(value, dict) else {}


def evidence_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("evidences")
    return value if isinstance(value, list) else []


def is_supporting(row: dict[str, Any], item: dict[str, Any]) -> bool:
    t = teacher(row)
    rank = item.get("rank")
    support_rank = t.get("supporting_evidence_rank")
    quote = str(t.get("supporting_evidence_quote", "")).strip()
    text = str(item.get("text", ""))
    return bool(item.get("answer_supported")) or (support_rank is not None and str(rank) == str(support_rank)) or bool(quote and quote in text)


def build(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    targets: dict[str, list[str]] = {}
    split_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    support_counts: Counter[str] = Counter()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        split_counts[str(row.get("split", ""))] += 1
        tier_counts[str(teacher(row).get("tier", ""))] += 1
        targets[sample_id] = []
        for item in evidence_items(row):
            if not isinstance(item, dict):
                continue
            rank = item.get("rank")
            doc_id = f"{sample_id}#e{rank}"
            supporting = is_supporting(row, item)
            if supporting:
                targets[sample_id].append(doc_id)
            support_counts[str(supporting)] += 1
            docs.append(
                {
                    "doc_id": doc_id,
                    "sample_id": sample_id,
                    "split": row.get("split"),
                    "rank": rank,
                    "text": str(item.get("text", "")),
                    "url": item.get("url", ""),
                    "domain": item.get("domain", ""),
                    "answer_supported": bool(item.get("answer_supported")),
                    "is_supporting_target": supporting,
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "visual_anchor": teacher(row).get("visual_anchor", ""),
                    "teacher_query": teacher(row).get("repaired_search_query", ""),
                }
            )
    stats = {
        "n_samples": len(rows),
        "n_docs": len(docs),
        "split_counts": dict(split_counts),
        "tier_counts": dict(tier_counts),
        "support_doc_counts": dict(support_counts),
        "samples_without_support_targets": sum(1 for ids in targets.values() if not ids),
    }
    return docs, targets, stats


def main() -> None:
    args = parse_args()
    package_dir = Path(args.package_dir).expanduser().resolve()
    main_path = Path(args.main_file)
    if not main_path.is_absolute():
        main_path = package_dir / main_path
    rows = load_jsonl(main_path)
    docs, targets, stats = build(rows)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "corpus.jsonl", docs)
    with (out_dir / "targets.json").open("w", encoding="utf-8") as f:
        json.dump(targets, f, ensure_ascii=False, indent=2)
    with (out_dir / "corpus_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
