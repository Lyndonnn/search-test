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
        default="data/pix2fact_dagig_train_AB_clean_split.jsonl",
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


def teacher(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("gpt54_teacher")
    return value if isinstance(value, dict) else {}


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
            ranked.append((doc["doc_id"], score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked


def evidence_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("evidences")
    return value if isinstance(value, list) else []


def corpus_text(row: dict[str, Any], item: dict[str, Any], mode: str) -> str:
    t = teacher(row)
    evidence_text = str(item.get("text", ""))
    supporting_quote = str(t.get("supporting_evidence_quote", ""))
    local_observation = str(t.get("local_observation") or row.get("qwen_local_observation", ""))
    if mode == "evidence_only":
        return " ".join([supporting_quote, evidence_text]).strip()
    if mode == "local_evidence":
        return " ".join(
            [
                local_observation,
                supporting_quote,
                evidence_text,
            ]
        )
    return " ".join(
        [
            str(row.get("question", "")),
            local_observation,
            supporting_quote,
            evidence_text,
        ]
    )


def supporting_doc_ids(row: dict[str, Any]) -> set[str]:
    sample_id = str(row.get("sample_id", ""))
    t = teacher(row)
    support_rank = t.get("supporting_evidence_rank")
    ids = set()
    for item in evidence_items(row):
        if not isinstance(item, dict):
            continue
        rank = item.get("rank")
        doc_id = f"{sample_id}#e{rank}"
        if bool(item.get("answer_supported")):
            ids.add(doc_id)
        elif support_rank is not None and str(rank) == str(support_rank):
            ids.add(doc_id)
    if not ids:
        quote = str(t.get("supporting_evidence_quote", "")).strip()
        for item in evidence_items(row):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", ""))
            if quote and quote in text:
                ids.add(f"{sample_id}#e{item.get('rank')}")
    return ids


def build_corpus(package_dir: Path, main_file: str, corpus_mode: str) -> tuple[BM25, dict[str, set[str]]]:
    path = Path(main_file)
    if not path.is_absolute():
        path = package_dir / path
    rows = load_jsonl(path)
    docs = []
    targets: dict[str, set[str]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        targets[sample_id] = supporting_doc_ids(row)
        for item in evidence_items(row):
            if not isinstance(item, dict):
                continue
            rank = item.get("rank")
            docs.append(
                {
                    "doc_id": f"{sample_id}#e{rank}",
                    "sample_id": sample_id,
                    "rank": str(rank),
                    "text": corpus_text(row, item, corpus_mode),
                }
            )
    return BM25(docs), targets


def load_search_rows(path: Path, dedupe: bool) -> list[dict[str, str]]:
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            has_chain_query = "search_prediction" in row
            if not has_chain_query and row.get("task_type") != "search_query":
                continue
            sample_id = str(row.get("sample_id", ""))
            pred = str(row.get("search_prediction") or row.get("prediction", ""))
            if dedupe:
                if sample_id in seen:
                    continue
                seen.add(sample_id)
            rows.append({"sample_id": sample_id, "prediction": pred, "target": str(row.get("search_target") or row.get("target", ""))})
    return rows


def reciprocal_rank(ranked: list[tuple[str, float]], target_ids: set[str]) -> float:
    for idx, (doc_id, _score) in enumerate(ranked, start=1):
        if doc_id in target_ids:
            return 1.0 / idx
    return 0.0


def target_rank(ranked: list[tuple[str, float]], target_ids: set[str]) -> int:
    for idx, (doc_id, _score) in enumerate(ranked, start=1):
        if doc_id in target_ids:
            return idx
    return len(ranked) + 1


def evaluate_file(path: Path, bm25: BM25, targets: dict[str, set[str]], dedupe: bool) -> dict[str, Any]:
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
        target_id = row["sample_id"]
        target_ids = targets.get(target_id, set())
        rank = target_rank(ranked, target_ids)
        target_ranks.append(rank)
        top_doc_ids = [doc_id for doc_id, _score in ranked]
        top1 += 1.0 if target_ids and bool(set(top_doc_ids[:1]) & target_ids) else 0.0
        top3 += 1.0 if target_ids and bool(set(top_doc_ids[:3]) & target_ids) else 0.0
        top5 += 1.0 if target_ids and bool(set(top_doc_ids[:5]) & target_ids) else 0.0
        mrr += reciprocal_rank(ranked, target_ids)
        ranked_scores = dict(ranked)
        target_score = max((ranked_scores.get(doc_id, 0.0) for doc_id in target_ids), default=0.0)
        best_wrong = max((score for doc_id, score in ranked if doc_id not in target_ids), default=0.0)
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
    bm25, targets = build_corpus(package_dir, args.main_file, args.corpus_mode)
    out_rows = [evaluate_file(Path(path), bm25, targets=targets, dedupe=args.dedupe) for path in args.details_csv]
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
