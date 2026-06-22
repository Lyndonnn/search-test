#!/usr/bin/env python3
"""Score Pix2Fact-DAGIG trajectories with baseline and DAG-IG rewards."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SEGMENTS = ["observe", "search_decision", "search", "evidence", "answer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples_jsonl", required=True, help="SFT/eval examples with target_segments and metadata.")
    parser.add_argument("--details_jsonl", default="", help="Optional model eval details JSONL from 03_eval_chain.py.")
    parser.add_argument("--corpus_jsonl", default="data/dagig_retrieval/corpus.jsonl")
    parser.add_argument("--targets_json", default="data/dagig_retrieval/targets.json")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--dedupe_sample_id", action="store_true")
    parser.add_argument("--allow_missing_details", action="store_true")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def token_f1(pred: str, target: str) -> float:
    pred_tokens = tokens(pred)
    target_tokens = tokens(target)
    if not pred_tokens or not target_tokens:
        return 0.0
    pc: dict[str, int] = defaultdict(int)
    tc: dict[str, int] = defaultdict(int)
    for tok in pred_tokens:
        pc[tok] += 1
    for tok in target_tokens:
        tc[tok] += 1
    common = sum(min(pc[tok], tc[tok]) for tok in pc)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_match(pred: str, target: str) -> bool:
    p = normalize(pred)
    t = normalize(target)
    return bool(p and t and (p in t or t in p))


def query_anchor_hit(query: str, anchor: str) -> bool:
    q_tokens = set(tokens(query))
    anchor_tokens = [tok for tok in tokens(anchor) if len(tok) > 1]
    return bool(anchor_tokens and any(tok in q_tokens for tok in anchor_tokens))


class BM25:
    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = docs
        self.doc_tokens = [tokens(str(doc.get("text", ""))) for doc in docs]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        n_docs = len(docs)
        self.idf = {tok: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5)) for tok, freq in df.items()}

    def rank(self, query: str) -> list[tuple[str, float]]:
        q_counts = Counter(tokens(query))
        ranked = []
        k1 = 1.2
        b = 0.75
        for doc, toks, dl in zip(self.docs, self.doc_tokens, self.doc_lens):
            tf = Counter(toks)
            score = 0.0
            for tok, qf in q_counts.items():
                if tok not in tf:
                    continue
                denom = tf[tok] + k1 * (1 - b + b * dl / max(self.avgdl, 1e-6))
                score += self.idf.get(tok, 0.0) * (tf[tok] * (k1 + 1) / denom) * qf
            ranked.append((str(doc["doc_id"]), score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked


def reciprocal_rank(ranked: list[tuple[str, float]], targets: set[str]) -> float:
    for idx, (doc_id, _score) in enumerate(ranked, start=1):
        if doc_id in targets:
            return 1.0 / idx
    return 0.0


def target_rank(ranked: list[tuple[str, float]], targets: set[str]) -> int:
    for idx, (doc_id, _score) in enumerate(ranked, start=1):
        if doc_id in targets:
            return idx
    return len(ranked) + 1


def details_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("sample_id", "")), str(row.get("variant", ""))


def prediction_segments(example: dict[str, Any], detail: dict[str, Any] | None) -> dict[str, str]:
    if detail:
        return {
            "observe": str(detail.get("observe_prediction", "")),
            "search_decision": str(detail.get("search_decision_prediction", "")),
            "search": str(detail.get("search_prediction", "")),
            "evidence": str(detail.get("evidence_prediction", "")),
            "answer": str(detail.get("answer_prediction", "")),
        }
    target = example.get("target_segments")
    if isinstance(target, dict):
        return {seg: str(target.get(seg, "")) for seg in SEGMENTS}
    return {seg: "" for seg in SEGMENTS}


def score_one(example: dict[str, Any], pred: dict[str, str], bm25: BM25, targets: dict[str, list[str]]) -> dict[str, Any]:
    sample_id = str(example.get("sample_id", ""))
    target = example.get("target_segments") if isinstance(example.get("target_segments"), dict) else {}
    target_ids = set(targets.get(sample_id, []))
    ranked = bm25.rank(pred["search"])
    rank = target_rank(ranked, target_ids)
    mrr = reciprocal_rank(ranked, target_ids)
    top_ids = [doc_id for doc_id, _score in ranked]
    r1 = 1.0 if target_ids and bool(set(top_ids[:1]) & target_ids) else 0.0
    r5 = 1.0 if target_ids and bool(set(top_ids[:5]) & target_ids) else 0.0

    answer_em = 1.0 if contains_match(pred["answer"], str(target.get("answer") or example.get("answer") or "")) else 0.0
    answer_f1 = token_f1(pred["answer"], str(target.get("answer") or example.get("answer") or ""))
    observe_f1 = token_f1(pred["observe"], str(target.get("observe", "")))
    query_f1 = token_f1(pred["search"], str(target.get("search", "")))
    evidence_f1 = token_f1(pred["evidence"], str(target.get("evidence", "")))
    anchor_hit = 1.0 if query_anchor_hit(pred["search"], str(example.get("visual_anchor", ""))) else 0.0
    valid_format = 1.0 if pred["observe"].strip() and pred["search"].strip() and pred["answer"].strip() else 0.0
    search_call = 1.0 if pred["search"].strip() else 0.0
    search_needed = 1.0 if bool(example.get("search_needed", True)) else 0.0
    unsupported = 1.0 if answer_em and evidence_f1 < 0.20 and r5 < 1.0 else 0.0
    spurious = 1.0 if answer_em and (anchor_hit < 1.0 or r5 < 1.0 or evidence_f1 < 0.20) else 0.0
    unnecessary_search = 1.0 if search_call and not search_needed else 0.0
    missing_search = 1.0 if search_needed and not search_call else 0.0

    locate_reward = 1.0
    observe_reward = observe_f1 if anchor_hit or observe_f1 >= 0.35 else 0.25 * observe_f1
    search_reward = 0.50 * anchor_hit + 0.30 * r5 + 0.20 * mrr
    evidence_reward = max(evidence_f1, r1)
    answer_reward = answer_f1 * (0.5 + 0.5 * max(evidence_f1, r5))
    cost = 0.05 * search_call + 0.20 * unnecessary_search + 0.30 * missing_search + 0.30 * spurious

    outcome_only = answer_em
    outcome_plus_search_penalty = max(0.0, answer_em - 0.05 * search_call - 0.25 * missing_search)
    generic_process = 0.20 * valid_format + 0.20 * observe_f1 + 0.20 * min(1.0, len(tokens(pred["search"])) / 8.0) + 0.20 * r5 + 0.20 * answer_f1
    text_ig = 0.35 * query_f1 + 0.35 * mrr + 0.30 * answer_f1
    dagig = max(
        0.0,
        0.10 * locate_reward
        + 0.20 * observe_reward
        + 0.25 * search_reward
        + 0.20 * evidence_reward
        + 0.25 * answer_reward
        - cost,
    )

    if answer_em and r5 < 1.0:
        failure = "spurious_or_unsupported_success"
    elif valid_format < 1.0:
        failure = "format_failure"
    elif observe_f1 < 0.25:
        failure = "observation_failure"
    elif anchor_hit < 1.0:
        failure = "query_anchor_failure"
    elif r5 < 1.0:
        failure = "retrieval_failure"
    elif answer_f1 < 0.5:
        failure = "answer_reasoning_failure"
    else:
        failure = "ok"

    return {
        "sample_id": sample_id,
        "variant": example.get("variant"),
        "split": example.get("split"),
        "observe_f1": observe_f1,
        "query_f1": query_f1,
        "evidence_f1": evidence_f1,
        "answer_f1": answer_f1,
        "answer_em": answer_em,
        "query_anchor_hit": anchor_hit,
        "retrieval_r1": r1,
        "retrieval_r5": r5,
        "retrieval_mrr": mrr,
        "supporting_evidence_rank": rank,
        "valid_format": valid_format,
        "search_call": search_call,
        "unnecessary_search": unnecessary_search,
        "missing_search": missing_search,
        "unsupported_answer": unsupported,
        "spurious_success": spurious,
        "reward_outcome_only": outcome_only,
        "reward_outcome_plus_search_penalty": outcome_plus_search_penalty,
        "reward_generic_process": generic_process,
        "reward_text_ig": text_ig,
        "reward_dagig": dagig,
        "failure_category": failure,
        "prediction": pred,
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("variant", "unknown"))].append(row)
        groups["all"].append(row)
    keys = [
        "answer_em",
        "answer_f1",
        "query_anchor_hit",
        "retrieval_r1",
        "retrieval_r5",
        "retrieval_mrr",
        "unsupported_answer",
        "spurious_success",
        "search_call",
        "reward_outcome_only",
        "reward_outcome_plus_search_penalty",
        "reward_generic_process",
        "reward_text_ig",
        "reward_dagig",
    ]
    out = []
    for variant, items in groups.items():
        row = {"variant": variant, "n": len(items)}
        for key in keys:
            row[key] = sum(float(item[key]) for item in items) / max(1, len(items))
        failures = Counter(str(item["failure_category"]) for item in items)
        for failure, count in failures.items():
            row[f"failure/{failure}"] = count / max(1, len(items))
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    examples = load_jsonl(args.examples_jsonl)
    if args.dedupe_sample_id:
        seen = set()
        deduped = []
        for ex in examples:
            sample_id = str(ex.get("sample_id", ""))
            if sample_id in seen:
                continue
            seen.add(sample_id)
            deduped.append(ex)
        examples = deduped
    details = {}
    if args.details_jsonl:
        for row in load_jsonl(args.details_jsonl):
            details[details_key(row)] = row
    corpus = load_jsonl(args.corpus_jsonl)
    with Path(args.targets_json).open("r", encoding="utf-8") as f:
        targets = json.load(f)
    bm25 = BM25(corpus)

    rows = []
    for ex in examples:
        detail = details.get(details_key(ex))
        if args.details_jsonl and detail is None and not args.allow_missing_details:
            continue
        pred = prediction_segments(ex, detail)
        rows.append(score_one(ex, pred, bm25, targets))
    if not rows:
        raise ValueError("No rows were scored. Check --details_jsonl variant/sample_id keys or pass --allow_missing_details.")

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = aggregate(rows)
    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in summary for key in row.keys()})
    priority = ["variant", "n"]
    fieldnames = priority + [key for key in fieldnames if key not in priority]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {out_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
