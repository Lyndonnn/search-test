#!/usr/bin/env python3
"""Evaluate generated search queries by evidence-retrieval utility.

This script uses the packaged Pix2Fact evidence texts as a small closed-book
retrieval corpus. It is not a replacement for web search, but it is a fast
diagnostic for whether a generated query points to the correct evidence item.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package_dir", required=True)
    parser.add_argument(
        "--main_file",
        default="dagig_relabel/qwen_dagig_reward_labeled_30_with_image_paths.jsonl",
        help="Path relative to package_dir unless absolute.",
    )
    parser.add_argument("--details_csv", action="append", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--dedupe", action="store_true", help="Deduplicate repeated dev rows by sample_id.")
    parser.add_argument(
        "--corpus_mode",
        choices=["all", "local_evidence", "evidence_only"],
        default="all",
        help=(
            "Fields used to build retrieval documents. Use evidence_only for a harder "
            "diagnostic that avoids question-token leakage."
        ),
    )
    return parser.parse_args()


def normalize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def method_from_path(path: Path) -> str:
    name = path.name
    for suffix in ["_eval_details.csv", "_details.csv", ".csv"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


class BM25:
    def __init__(self, docs: list[dict[str, str]]):
        self.docs = docs
        self.doc_tokens = [normalize(doc["text"]) for doc in docs]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        n_docs = len(docs)
        self.idf = {tok: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5)) for tok, freq in df.items()}

    def rank(self, query: str) -> list[tuple[str, float]]:
        q = normalize(query)
        q_counts = Counter(q)
        ranked = []
        k1 = 1.2
        b = 0.75
        for doc, toks, dl in zip(self.docs, self.doc_tokens, self.doc_lens):
            tf = Counter(toks)
            score = 0.0
            for tok, qf in q_counts.items():
                if tok not in tf:
                    continue
                idf = self.idf.get(tok, 0.0)
                denom = tf[tok] + k1 * (1 - b + b * dl / max(self.avgdl, 1e-6))
                score += idf * (tf[tok] * (k1 + 1) / denom) * qf
            ranked.append((doc["sample_id"], score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked


def corpus_text(row: dict[str, Any], mode: str) -> str:
    if mode == "evidence_only":
        return str(row.get("selected_evidence_text", ""))
    if mode == "local_evidence":
        return " ".join(
            [
                str(row.get("qwen_local_observation", "")),
                str(row.get("selected_evidence_text", "")),
            ]
        )
    return " ".join(
        [
            str(row.get("question", "")),
            str(row.get("qwen_local_observation", "")),
            str(row.get("selected_evidence_text", "")),
        ]
    )


def build_corpus(package_dir: Path, main_file: str, corpus_mode: str) -> tuple[BM25, dict[str, dict[str, Any]]]:
    path = Path(main_file)
    if not path.is_absolute():
        path = package_dir / path
    rows = load_jsonl(path)
    by_id = {str(r["sample_id"]): r for r in rows}
    docs = []
    for row in rows:
        text = corpus_text(row, corpus_mode)
        docs.append({"sample_id": str(row["sample_id"]), "text": text})
    return BM25(docs), by_id


def load_search_rows(path: Path, dedupe: bool) -> list[dict[str, str]]:
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("task_type") != "search_query":
                continue
            sample_id = str(row.get("sample_id", ""))
            pred = str(row.get("prediction", ""))
            if dedupe:
                if sample_id in seen:
                    continue
                seen.add(sample_id)
            rows.append({"sample_id": sample_id, "prediction": pred, "target": str(row.get("target", ""))})
    return rows


def reciprocal_rank(ranked: list[tuple[str, float]], target_id: str) -> float:
    for idx, (sample_id, _score) in enumerate(ranked, start=1):
        if sample_id == target_id:
            return 1.0 / idx
    return 0.0


def target_rank(ranked: list[tuple[str, float]], target_id: str) -> int:
    for idx, (sample_id, _score) in enumerate(ranked, start=1):
        if sample_id == target_id:
            return idx
    return len(ranked) + 1


def evaluate_file(path: Path, bm25: BM25, dedupe: bool) -> dict[str, Any]:
    rows = load_search_rows(path, dedupe=dedupe)
    if not rows:
        raise ValueError(f"No search_query rows found in {path}")
    top1 = top3 = top5 = mrr = 0.0
    nonempty = 0.0
    score_at_target = []
    target_ranks = []
    score_margins = []
    for row in rows:
        pred = row["prediction"].strip()
        nonempty += 1.0 if pred else 0.0
        ranked = bm25.rank(pred)
        ids = [sample_id for sample_id, _score in ranked]
        target_id = row["sample_id"]
        rank = target_rank(ranked, target_id)
        target_ranks.append(rank)
        top1 += 1.0 if ids[:1] == [target_id] else 0.0
        top3 += 1.0 if target_id in ids[:3] else 0.0
        top5 += 1.0 if target_id in ids[:5] else 0.0
        mrr += reciprocal_rank(ranked, target_id)
        ranked_scores = dict(ranked)
        target_score = ranked_scores.get(target_id, 0.0)
        best_wrong = max((score for sample_id, score in ranked if sample_id != target_id), default=0.0)
        score_at_target.append(target_score)
        score_margins.append(target_score - best_wrong)
    n = len(rows)
    return {
        "method": method_from_path(path),
        "details_csv": str(path),
        "n": n,
        "query_nonempty_rate": nonempty / n,
        "retrieval_top1": top1 / n,
        "retrieval_top3": top3 / n,
        "retrieval_top5": top5 / n,
        "retrieval_mrr": mrr / n,
        "target_rank_mean": sum(target_ranks) / n,
        "target_score_mean": sum(score_at_target) / n,
        "target_score_margin_mean": sum(score_margins) / n,
    }


def main() -> None:
    args = parse_args()
    package_dir = Path(args.package_dir).expanduser().resolve()
    bm25, _by_id = build_corpus(package_dir, args.main_file, args.corpus_mode)
    out_rows = [evaluate_file(Path(path), bm25, dedupe=args.dedupe) for path in args.details_csv]
    for row in out_rows:
        row["corpus_mode"] = args.corpus_mode
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {out_path}")
    for row in out_rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
