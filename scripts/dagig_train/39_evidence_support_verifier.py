#!/usr/bin/env python3
"""Lightweight evidence support verifier for DAG-IG v2."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from grounded_pipeline_utils import load_jsonl, md_table, token_f1, tokenize, write_jsonl


SPLITS = ["train", "dev", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/dagig_rn03_10_grounded_rl")
    parser.add_argument("--out_dir", default="results/dagig_rn03_10_counterfactual_dagig_v2")
    parser.add_argument("--mode", choices=["heuristic", "hybrid", "local_llm"], default="hybrid")
    return parser.parse_args()


def normalize(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def answer_present(answer: Any, evidence: Any) -> bool:
    ans = normalize(answer)
    body = normalize(evidence)
    compact = ans.replace(" ", "")
    min_len = 2 if compact.isdigit() else 3
    return bool(ans and len(compact) >= min_len and ans in body)


def entity_match(question: Any, evidence: Any) -> bool:
    q_tokens = [tok for tok in tokenize(str(question or "")) if len(tok) >= 5]
    e_tokens = set(tokenize(str(evidence or "")))
    return bool(q_tokens and any(tok in e_tokens for tok in q_tokens))


def contradiction_flag(candidate_answer: Any, evidence: Any, gold_answer: Any) -> bool:
    cand = normalize(candidate_answer)
    gold = normalize(gold_answer)
    if not cand or not gold or cand == gold:
        return False
    return answer_present(candidate_answer, evidence) and not answer_present(gold_answer, evidence)


def verify_evidence(
    question: Any,
    candidate_answer: Any,
    evidence_text: Any,
    gold_answer: Any,
    mode: str = "hybrid",
) -> dict[str, Any]:
    evidence = str(evidence_text or "")
    answer_string_present = answer_present(candidate_answer, evidence) or answer_present(gold_answer, evidence)
    ent = entity_match(question, evidence)
    contradiction = contradiction_flag(candidate_answer, evidence, gold_answer)
    answer_overlap = max(token_f1(str(candidate_answer or ""), evidence), token_f1(str(gold_answer or ""), evidence))
    length_ok = len(tokenize(evidence)) >= 3
    score = 0.0
    score += 0.45 if answer_string_present else 0.0
    score += 0.20 if ent else 0.0
    score += 0.20 * min(1.0, answer_overlap * 2.0)
    score += 0.15 if length_ok else 0.0
    if contradiction:
        score -= 0.50
    score = max(0.0, min(1.0, score))
    if mode == "local_llm":
        reason = "local_llm_requested_but_not_configured_fallback_to_hybrid"
    elif mode == "heuristic":
        reason = "heuristic_overlap_answer_entity"
    else:
        reason = "hybrid_heuristic_answer_entity_overlap"
    return {
        "supports_answer": bool(score >= 0.50 and not contradiction),
        "support_score": float(score),
        "answer_string_present": bool(answer_string_present),
        "entity_match": bool(ent),
        "contradiction_flag": bool(contradiction),
        "reason": reason,
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for split in SPLITS:
        rows = load_jsonl(Path(args.data_dir) / f"grounded_rl_{split}.jsonl")
        out = []
        for row in rows:
            result = verify_evidence(
                row.get("question"),
                row.get("gold_answer") or row.get("answer"),
                row.get("teacher_evidence_text") or row.get("evidence"),
                row.get("gold_answer") or row.get("answer"),
                args.mode,
            )
            out.append({"sample_id": row.get("sample_id"), "split": split, **result})
        write_jsonl(out_dir / f"evidence_verifier_{split}.jsonl", out)
        summary.append(
            {
                "split": split,
                "n": len(out),
                "supports_answer_rate": mean([float(row["supports_answer"]) for row in out]) if out else 0.0,
                "mean_support_score": mean([float(row["support_score"]) for row in out]) if out else 0.0,
            }
        )
    (out_dir / "evidence_verifier_summary.md").write_text(
        "# Evidence Verifier Summary\n\n" + md_table(summary) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
