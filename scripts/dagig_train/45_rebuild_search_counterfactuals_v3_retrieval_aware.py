#!/usr/bin/env python3
"""Build retrieval-aware adversarial search counterfactuals for DAG-IG v3."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from grounded_pipeline_utils import load_jsonl, md_table, tokenize, write_csv, write_jsonl


SCORER_PATH = Path(__file__).with_name("26_score_grounded_rollouts.py")
SPEC = importlib.util.spec_from_file_location("grounded_rollout_scorer", SCORER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import scorer from {SCORER_PATH}")
SCORER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER_MODULE)
BM25 = SCORER_MODULE.BM25
load_corpus = SCORER_MODULE.load_corpus
load_targets = SCORER_MODULE.load_targets


SPLITS = ["train", "dev", "test"]
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "what",
    "which",
    "how",
    "many",
    "number",
    "contact",
    "official",
    "website",
    "information",
    "count",
    "address",
    "phone",
    "email",
}
TASK_PATTERNS = [
    ("repair contact number", ["repair", "contact", "phone", "number"]),
    ("contact phone number", ["contact", "phone", "number"]),
    ("email address", ["email", "address"]),
    ("store address", ["store", "address"]),
    ("branch count", ["branch", "branches", "count"]),
    ("official report count", ["report", "reports", "count"]),
    ("corporate registration number", ["registration", "number"]),
    ("price information", ["price", "cost"]),
    ("release date", ["release", "date"]),
    ("opening hours", ["hours", "opening"]),
    ("official website information", ["official", "website", "information"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/dagig_rn03_10_grounded_rl")
    parser.add_argument("--out_dir", default="data/dagig_rn03_10_counterfactuals_v3")
    parser.add_argument("--result_dir", default="results/dagig_rn03_10_counterfactual_dagig_v3")
    parser.add_argument("--corpus_jsonl", default="data/dagig_rn03_10_retrieval/corpus.jsonl")
    parser.add_argument("--targets_json", default="data/dagig_rn03_10_retrieval/targets.json")
    parser.add_argument("--min_valid_rate", type=float, default=0.60)
    parser.add_argument("--min_real_beats_rate", type=float, default=0.60)
    parser.add_argument("--min_support_drop_rate", type=float, default=0.40)
    parser.add_argument("--max_duplicate_rate", type=float, default=0.10)
    parser.add_argument("--no_fail_on_weak", action="store_true")
    return parser.parse_args()


def clean_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def tokens_no_stop(text: Any) -> list[str]:
    return [tok for tok in tokenize(str(text or "")) if tok not in STOPWORDS and len(tok) >= 3]


def remove_phrases(text: str, phrases: list[str]) -> str:
    out = str(text or "")
    for phrase in sorted({p for p in phrases if p}, key=len, reverse=True):
        out = re.sub(re.escape(phrase), " ", out, flags=re.I)
        for tok in sorted(set(tokenize(phrase)), key=len, reverse=True):
            if len(tok) >= 2:
                out = re.sub(rf"\b{re.escape(tok)}\b", " ", out, flags=re.I)
    return clean_space(out).strip(" ,;:-")


def capitalized_phrases(text: str) -> list[str]:
    matches = re.findall(r"(?:[A-Z][A-Za-z0-9&'.-]+(?:\s+|$)){1,5}", str(text or ""))
    return [clean_space(m) for m in matches if len(clean_space(m)) >= 2]


def answer_like_strings(row: dict[str, Any]) -> list[str]:
    out = [str(row.get("gold_answer") or ""), str(row.get("answer") or "")]
    query = str(row.get("teacher_search_query") or "")
    out.extend(re.findall(r"\b\d[\d\s+()./-]{1,}\b", query))
    return [clean_space(x) for x in out if clean_space(x)]


def task_generic(row: dict[str, Any]) -> str:
    text = " ".join([str(row.get("question") or ""), str(row.get("teacher_search_query") or "")]).lower()
    for label, keys in TASK_PATTERNS:
        if any(key in text for key in keys):
            return label
    query_tokens = tokens_no_stop(row.get("teacher_search_query"))
    generic = [tok for tok in query_tokens if tok in {"report", "reports", "subsidiaries", "repair", "contact", "phone", "branch", "branches", "store", "address", "email", "price", "date", "count"}]
    return " ".join(generic[:4]) or "official website information"


def task_tail(row: dict[str, Any]) -> str:
    query = str(row.get("teacher_search_query") or "")
    anchors = [
        str(row.get("semantic_anchor") or ""),
        str(row.get("visual_anchor") or ""),
        str(row.get("teacher_ground_expression") or ""),
        *capitalized_phrases(query),
        *answer_like_strings(row),
    ]
    strict = remove_phrases(query, anchors)
    toks = [tok for tok in tokens_no_stop(strict) if not tok.isdigit()]
    if len(toks) >= 2:
        return " ".join(toks[:8])
    return task_generic(row)


def task_signature(row: dict[str, Any]) -> str:
    generic = task_generic(row)
    return generic.split()[0] if generic else "generic"


def entity_removed_strict(row: dict[str, Any]) -> str:
    query = str(row.get("teacher_search_query") or "")
    question = str(row.get("question") or "")
    anchors = [
        str(row.get("semantic_anchor") or ""),
        str(row.get("visual_anchor") or ""),
        str(row.get("teacher_ground_expression") or ""),
        *capitalized_phrases(query),
        *capitalized_phrases(question),
        *answer_like_strings(row),
    ]
    removed = remove_phrases(query, anchors)
    toks = [tok for tok in tokens_no_stop(removed) if not tok.isdigit()]
    # Keep only task words; if disambiguators remain, fall back to the strict generic task.
    allowed = {"report", "reports", "sustainability", "repair", "contact", "phone", "number", "faq", "sms", "notification", "notifications", "branch", "branches", "loan", "specialist", "email", "address", "price", "date", "registration", "count", "store", "official"}
    kept = [tok for tok in toks if tok in allowed]
    return " ".join(kept[:6]) or task_generic(row)


def doc_entity(doc: dict[str, Any]) -> str:
    for key in ("visual_anchor", "teacher_query", "domain", "url"):
        text = str(doc.get(key) or "")
        if key == "domain" and text:
            return text.split(".")[0]
        caps = capitalized_phrases(text)
        if caps:
            return caps[0]
    toks = tokens_no_stop(doc.get("text"))
    return " ".join(toks[:3]) if toks else "different organization"


class Retrieval:
    def __init__(self, corpus_jsonl: str, targets_json: str):
        self.docs = load_corpus(corpus_jsonl)
        self.bm25 = BM25(self.docs)
        self.targets = {k: set(v) for k, v in load_targets(targets_json).items()}
        self.doc_by_id = {str(doc.get("doc_id")): doc for doc in self.docs}

    def rank(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        return self.bm25.rank(query, k=k or len(self.docs))

    def evaluate(self, sample_id: str, query: str, anchor: str = "", answer: str = "") -> dict[str, Any]:
        ranked = self.rank(query, k=len(self.docs))
        targets = self.targets.get(sample_id, set())
        target_rank = len(ranked) + 1
        target_score = 0.0
        for idx, (doc_id, score) in enumerate(ranked, start=1):
            if doc_id in targets:
                target_rank = idx
                target_score = float(score)
                break
        top_ids = [doc_id for doc_id, _score in ranked[:5]]
        top_scores = [float(score) for _doc_id, score in ranked[:5]]
        target_doc_in_top5 = bool(set(top_ids) & targets)
        answer_norm = normalize(answer)
        anchor_tokens = set(tokens_no_stop(anchor))
        top_text = "\n".join(str(self.doc_by_id.get(doc_id, {}).get("text") or "") for doc_id in top_ids)
        return {
            "target_rank": target_rank,
            "support@1": bool(ranked and ranked[0][0] in targets),
            "support@5": target_doc_in_top5,
            "MRR": 1.0 / target_rank if target_rank <= len(ranked) else 0.0,
            "target_doc_in_top5": target_doc_in_top5,
            "target_score": target_score,
            "retrieval_score": retrieval_score(target_rank, target_doc_in_top5, ranked[0][0] in targets if ranked else False),
            "top_ids": top_ids,
            "top_scores": top_scores,
            "answer_string_hit": bool(answer_norm and answer_norm in normalize(top_text)),
            "entity_anchor_hit": bool(anchor_tokens and anchor_tokens & set(tokenize(top_text))),
        }

    def top_non_target_docs(self, sample_id: str, query: str, k: int = 20) -> list[dict[str, Any]]:
        targets = self.targets.get(sample_id, set())
        docs = []
        for doc_id, score in self.rank(query, k=k):
            if doc_id not in targets:
                doc = self.doc_by_id.get(doc_id)
                if doc:
                    docs.append(doc | {"bm25_score": score})
        return docs


def retrieval_score(rank: int, support5: bool, support1: bool) -> float:
    if not support5:
        return 0.0 if rank > 50 else 0.30 * (1.0 / max(rank, 1))
    return 0.45 + 0.25 * float(support1) + 0.30 * (1.0 / max(rank, 1))


def valid_cf(real: dict[str, Any], cf: dict[str, Any]) -> tuple[bool, str]:
    if cf["top_ids"] == real["top_ids"]:
        return False, "same_top5"
    if int(cf["target_rank"]) <= int(real["target_rank"]) and float(cf["retrieval_score"]) >= float(real["retrieval_score"]):
        return False, "target_rank_not_lower"
    if real["support@5"] and not cf["support@5"]:
        return True, "support5_drop"
    if int(real["target_rank"]) <= 5 and int(cf["target_rank"]) > 5:
        return True, "rank_out_of_top5"
    if int(cf["target_rank"]) - int(real["target_rank"]) >= 5:
        return True, "rank_delta_ge_5"
    if float(real["retrieval_score"]) - float(cf["retrieval_score"]) >= 0.25:
        return True, "score_drop_ge_0_25"
    return False, "insufficient_retrieval_drop"


def candidate_queries(row: dict[str, Any], rows: list[dict[str, Any]], retrieval: Retrieval) -> list[dict[str, Any]]:
    sample_id = str(row.get("sample_id"))
    real = clean_space(row.get("teacher_search_query") or (row.get("target_segments") or {}).get("search"))
    anchor = clean_space(row.get("semantic_anchor") or row.get("visual_anchor"))
    tail = task_tail(row)
    task = task_generic(row)
    candidates: list[dict[str, Any]] = [
        {"type": "entity_removed_strict", "text": entity_removed_strict(row), "source": "rule"},
        {"type": "task_generic_strict", "text": task, "source": "rule"},
    ]

    same_task = [other for other in rows if other.get("sample_id") != row.get("sample_id") and task_signature(other) == task_signature(row)]
    for other in same_task[:30]:
        other_anchor = clean_space(other.get("semantic_anchor") or other.get("visual_anchor"))
        if other_anchor and normalize(other_anchor) != normalize(anchor):
            candidates.append({"type": "anchor_swapped_same_task", "text": f"{other_anchor} {tail}", "source": other.get("sample_id")})

    # Corpus-aware hard negatives: use docs that real query already likes but are not target docs.
    for doc in retrieval.top_non_target_docs(sample_id, real, k=25):
        entity = doc_entity(doc)
        if entity:
            candidates.append({"type": "top_confuser_query", "text": f"{entity} {tail}", "source": doc.get("doc_id")})
            candidates.append({"type": "retrieval_hard_negative_entity", "text": f"{entity} {task}", "source": doc.get("doc_id")})

    # Add supporting docs from same-task samples as stronger retrieval-aware confusers.
    for other in same_task[:50]:
        other_anchor = clean_space(other.get("semantic_anchor") or other.get("visual_anchor"))
        if other_anchor and normalize(other_anchor) != normalize(anchor):
            candidates.append({"type": "retrieval_hard_negative_entity", "text": f"{other_anchor} {task}", "source": other.get("sample_id")})

    deduped = []
    seen = set()
    for cand in candidates:
        text = clean_space(cand.get("text"))
        key = normalize(text)
        if not text or key == normalize(real) or key in seen:
            continue
        seen.add(key)
        deduped.append(cand | {"text": text})
    return deduped


def best_valid(valids: list[dict[str, Any]], real: dict[str, Any]) -> dict[str, Any] | None:
    if not valids:
        return None
    # Strong hard negative: high wrong-doc score but target retrieval much worse.
    def key(item: dict[str, Any]) -> tuple[float, float, float]:
        ret = item["retrieval"]
        rank_delta = float(ret["target_rank"]) - float(real["target_rank"])
        score_drop = float(real["retrieval_score"]) - float(ret["retrieval_score"])
        wrong_top_score = float(ret["top_scores"][0]) if ret.get("top_scores") else 0.0
        return (score_drop + 0.02 * min(rank_delta, 20.0), wrong_top_score, -float(ret["target_rank"]))

    return max(valids, key=key)


def build_split(split: str, rows: list[dict[str, Any]], retrieval: Retrieval) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        sample_id = str(row.get("sample_id"))
        real_query = clean_space(row.get("teacher_search_query") or (row.get("target_segments") or {}).get("search"))
        anchor = clean_space(row.get("semantic_anchor") or row.get("visual_anchor"))
        answer = clean_space(row.get("gold_answer") or row.get("answer"))
        real = retrieval.evaluate(sample_id, real_query, anchor=anchor, answer=answer)
        candidates = []
        valid_candidates = []
        for cand in candidate_queries(row, rows, retrieval):
            metrics = retrieval.evaluate(sample_id, cand["text"], anchor=anchor, answer=answer)
            is_valid, reason = valid_cf(real, metrics)
            rank_delta = int(metrics["target_rank"]) - int(real["target_rank"])
            score_drop = float(real["retrieval_score"]) - float(metrics["retrieval_score"])
            item = {
                **cand,
                "retrieval": metrics,
                "valid": bool(is_valid),
                "valid_reason": reason,
                "rank_delta": rank_delta,
                "score_drop": score_drop,
                "support_drop": bool(real["support@5"] and not metrics["support@5"]),
            }
            candidates.append(item)
            if is_valid:
                valid_candidates.append(item)
        best = best_valid(valid_candidates, real)
        best_ret = best.get("retrieval") if best else {}
        output.append(
            {
                "sample_id": sample_id,
                "split": split,
                "real_query": real_query,
                "real_retrieval": real,
                "candidate_counterfactuals": candidates,
                "valid_counterfactuals": valid_candidates,
                "best_valid_cf": best or None,
                "search_cf_valid": best is not None,
                "rank_delta": (int(best_ret.get("target_rank")) - int(real["target_rank"])) if best_ret else None,
                "support_drop": bool(best_ret and real["support@5"] and not best_ret.get("support@5")),
                "real_beats_best_valid_cf": bool(best_ret and float(real["retrieval_score"]) > float(best_ret.get("retrieval_score") or 0.0)),
                "reason": best.get("valid_reason") if best else "no_valid_counterfactual",
            }
        )
    return output


def diagnostics(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    n = len(rows)
    valid_rows = [row for row in rows if row.get("search_cf_valid")]
    all_candidates = [cand for row in rows for cand in row.get("candidate_counterfactuals", [])]
    duplicate_count = 0
    total_candidate_slots = 0
    invalid_same = 0
    for row in rows:
        texts = [normalize(cand.get("text")) for cand in row.get("candidate_counterfactuals", [])]
        total_candidate_slots += len(texts)
        duplicate_count += len(texts) - len(set(texts))
        invalid_same += sum(1 for cand in row.get("candidate_counterfactuals", []) if cand.get("valid_reason") == "same_top5")
    rank_deltas = [float(row["rank_delta"]) for row in valid_rows if row.get("rank_delta") is not None and math.isfinite(float(row["rank_delta"]))]
    return {
        "split": split,
        "n": n,
        "valid_search_cf_rate": len(valid_rows) / max(1, n),
        "real_beats_best_valid_cf_rate": sum(1 for row in rows if row.get("real_beats_best_valid_cf")) / max(1, n),
        "mean_rank_delta": mean(rank_deltas) if rank_deltas else 0.0,
        "support@5_drop_rate": sum(1 for row in rows if row.get("support_drop")) / max(1, n),
        "duplicate_cf_rate": duplicate_count / max(1, total_candidate_slots),
        "invalid_same_retrieval_rate": invalid_same / max(1, len(all_candidates)),
        "mean_candidates": len(all_candidates) / max(1, n),
    }


def passes(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        float(row["valid_search_cf_rate"]) >= args.min_valid_rate
        and float(row["real_beats_best_valid_cf_rate"]) >= args.min_real_beats_rate
        and float(row["support@5_drop_rate"]) >= args.min_support_drop_rate
        and float(row["duplicate_cf_rate"]) <= args.max_duplicate_rate
    )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    result_dir = Path(args.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    retrieval = Retrieval(args.corpus_jsonl, args.targets_json)
    summary = []
    for split in SPLITS:
        rows = load_jsonl(Path(args.data_dir) / f"grounded_rl_{split}.jsonl")
        output = build_split(split, rows, retrieval)
        write_jsonl(out_dir / f"search_counterfactuals_v3_{split}.jsonl", output)
        summary.append(diagnostics(output, split))
    for row in summary:
        row["passes_thresholds"] = passes(row, args)
    write_csv(result_dir / "search_counterfactual_v3_diagnostics.csv", summary)
    threshold_rows = [
        {"metric": "valid_search_cf_rate", "threshold": args.min_valid_rate},
        {"metric": "real_beats_best_valid_cf_rate", "threshold": args.min_real_beats_rate},
        {"metric": "support@5_drop_rate", "threshold": args.min_support_drop_rate},
        {"metric": "duplicate_cf_rate", "threshold": f"<= {args.max_duplicate_rate}"},
    ]
    md = [
        "# Search Counterfactual v3 Diagnostics",
        "",
        md_table(summary),
        "",
        "## Thresholds",
        "",
        md_table(threshold_rows),
    ]
    (result_dir / "search_counterfactual_v3_diagnostics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.no_fail_on_weak and any(not row["passes_thresholds"] for row in summary):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
