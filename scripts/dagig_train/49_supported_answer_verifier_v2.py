#!/usr/bin/env python3
"""Deterministic supported-answer verifier for DAG-IG v3.1."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from statistics import mean
from typing import Any

from grounded_pipeline_utils import extract_segments, load_jsonl, md_table, token_f1, tokenize, write_jsonl


SPLITS = ["train", "dev", "test"]
TAGS = ["ground", "observe", "search_decision", "search", "evidence", "answer"]
MONTHS = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored_dir", default="results/dagig_rn03_10_counterfactual_dagig_v3")
    parser.add_argument("--rl_data_dir", default="data/dagig_rn03_10_grounded_rl")
    parser.add_argument("--out_dir", default="results/dagig_rn03_10_counterfactual_dagig_v31")
    parser.add_argument("--mode", choices=["hybrid", "hybrid_llm"], default="hybrid")
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    return parser.parse_args()


def clean_text(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).strip()


def normalize_text(text: Any) -> str:
    text = clean_text(text).lower()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text).strip()


def compact_alnum(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_text(text).lower())


def digits_only(text: Any) -> str:
    return re.sub(r"\D+", "", clean_text(text))


def extract_phone_candidates(text: Any) -> list[str]:
    raw = clean_text(text)
    candidates = []
    for match in re.finditer(r"(?:\+?\d[\d\s()./-]{5,}\d)", raw):
        digits = digits_only(match.group(0))
        if len(digits) >= 7:
            candidates.append(digits)
    return sorted(set(candidates), key=len, reverse=True)


def extract_emails(text: Any) -> list[str]:
    return sorted(set(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", clean_text(text).lower())))


def extract_prices(text: Any) -> list[str]:
    raw = clean_text(text)
    out = []
    for match in re.finditer(r"(?:[$€£¥]\s*)?\d+(?:[,.]\d{3})*(?:\.\d+)?(?:\s*(?:usd|eur|gbp|dollars?|euros?|pounds?))?", raw, flags=re.I):
        token = match.group(0)
        if any(sym in token.lower() for sym in ["$", "€", "£", "¥", "usd", "eur", "gbp", "dollar", "euro", "pound"]):
            out.append(re.sub(r"[, ]+", "", token.lower()))
    return sorted(set(out))


def extract_numbers(text: Any) -> list[str]:
    return sorted(set(re.findall(r"\d+(?:[,.]\d+)?", clean_text(text))))


def extract_dates(text: Any) -> list[str]:
    raw = clean_text(text).lower()
    dates = set()
    for y, m, d in re.findall(r"\b(20\d{2}|19\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", raw):
        dates.add(f"{int(y):04d}-{int(m):02d}-{int(d):02d}")
    for d, m, y in re.findall(r"\b(\d{1,2})[-/](\d{1,2})[-/](20\d{2}|19\d{2})\b", raw):
        dates.add(f"{int(y):04d}-{int(m):02d}-{int(d):02d}")
    month_re = "|".join(sorted(MONTHS, key=len, reverse=True))
    for mon, d, y in re.findall(rf"\b({month_re})\s+(\d{{1,2}}),?\s+(20\d{{2}}|19\d{{2}})\b", raw):
        dates.add(f"{int(y):04d}-{MONTHS[mon]}-{int(d):02d}")
    for d, mon, y in re.findall(rf"\b(\d{{1,2}})\s+({month_re})\s+(20\d{{2}}|19\d{{2}})\b", raw):
        dates.add(f"{int(y):04d}-{MONTHS[mon]}-{int(d):02d}")
    return sorted(dates)


def infer_answer_type(question: Any, gold_answer: Any, predicted_answer: Any = "") -> str:
    text = normalize_text(f"{question} {gold_answer} {predicted_answer}")
    gold = clean_text(gold_answer)
    if extract_emails(gold):
        return "email"
    if "email" in text or "e mail" in text:
        return "email"
    if extract_phone_candidates(gold) or any(k in text for k in ["phone", "telephone", "contact number", "call", "电话号码"]):
        return "phone"
    if extract_prices(gold) or any(k in text for k in ["price", "cost", "fee", "多少钱"]):
        return "price"
    if extract_dates(gold) or any(k in text for k in ["date", "when", "release", "founded", "opened"]):
        return "date"
    if any(k in text for k in ["address", "where", "location", "store", "branch"]):
        return "address"
    if re.fullmatch(r"\s*\d+(?:[,.]\d+)?\s*", gold) and any(k in text for k in ["how many", "count", "number of", "branches", "reports"]):
        return "count"
    if re.search(r"\b(?:registration|license|company id|ein|vat|number)\b", text) and re.search(r"\d", gold):
        return "identifier"
    return "text"


def phone_equivalent(a: Any, b: Any) -> bool:
    phones_a = extract_phone_candidates(a)
    phones_b = extract_phone_candidates(b)
    for pa in phones_a:
        for pb in phones_b:
            if pa == pb:
                return True
            min_suffix = 8 if min(len(pa), len(pb)) >= 8 else 7
            if len(pa) >= min_suffix and len(pb) >= min_suffix and pa[-min_suffix:] == pb[-min_suffix:]:
                return True
    return False


def numeric_equivalent(a: Any, b: Any) -> bool:
    nums_a = [x.replace(",", "") for x in extract_numbers(a)]
    nums_b = [x.replace(",", "") for x in extract_numbers(b)]
    return bool(nums_a and nums_b and set(nums_a) & set(nums_b))


def date_equivalent(a: Any, b: Any) -> bool:
    da = set(extract_dates(a))
    db = set(extract_dates(b))
    if da and db:
        return bool(da & db)
    return False


def email_equivalent(a: Any, b: Any) -> bool:
    ea = set(extract_emails(a))
    eb = set(extract_emails(b))
    return bool(ea and eb and ea & eb)


def text_equivalent(a: Any, b: Any) -> bool:
    na = normalize_text(a)
    nb = normalize_text(b)
    ca = compact_alnum(a)
    cb = compact_alnum(b)
    if not na or not nb:
        return False
    if na == nb or ca == cb:
        return True
    if len(ca) >= 4 and len(cb) >= 4 and (ca in cb or cb in ca):
        return True
    return token_f1(str(a), str(b)) >= 0.80


def answer_equivalence(predicted: Any, gold: Any, question: Any = "") -> dict[str, Any]:
    answer_type = infer_answer_type(question, gold, predicted)
    predicted_text = clean_text(predicted)
    gold_text = clean_text(gold)
    type_valid = bool(predicted_text)
    equivalent = False
    reason = "text_overlap"
    if answer_type == "phone":
        type_valid = bool(extract_phone_candidates(predicted_text))
        equivalent = phone_equivalent(predicted_text, gold_text)
        reason = "phone_equivalence"
    elif answer_type == "email":
        type_valid = bool(extract_emails(predicted_text))
        equivalent = email_equivalent(predicted_text, gold_text)
        reason = "email_equivalence"
    elif answer_type in {"count", "identifier", "price"}:
        type_valid = bool(re.search(r"\d", predicted_text))
        equivalent = numeric_equivalent(predicted_text, gold_text) or text_equivalent(predicted_text, gold_text)
        reason = f"{answer_type}_numeric_equivalence"
    elif answer_type == "date":
        type_valid = bool(extract_dates(predicted_text) or re.search(r"\d", predicted_text))
        equivalent = date_equivalent(predicted_text, gold_text) or text_equivalent(predicted_text, gold_text)
        reason = "date_equivalence"
    else:
        equivalent = text_equivalent(predicted_text, gold_text)
    f1 = 1.0 if equivalent else token_f1(predicted_text, gold_text)
    return {
        "answer_type": answer_type,
        "answer_type_valid": bool(type_valid),
        "answer_correct": bool(equivalent),
        "answer_em": 1.0 if equivalent else 0.0,
        "answer_f1": float(f1),
        "answer_match_reason": reason,
    }


def answer_appears_in_text(answer: Any, text: Any, question: Any = "") -> bool:
    answer_type = infer_answer_type(question, answer)
    if not clean_text(answer) or not clean_text(text):
        return False
    if answer_type == "phone":
        return phone_equivalent(answer, text)
    if answer_type == "email":
        return email_equivalent(answer, text)
    if answer_type in {"count", "identifier", "price"}:
        return numeric_equivalent(answer, text) or compact_alnum(answer) in compact_alnum(text)
    if answer_type == "date":
        return date_equivalent(answer, text) or compact_alnum(answer) in compact_alnum(text)
    ans = normalize_text(answer)
    body = normalize_text(text)
    cans = compact_alnum(answer)
    cbody = compact_alnum(text)
    return bool((ans and ans in body) or (len(cans) >= 4 and cans in cbody) or token_f1(str(answer), str(text)) >= 0.65)


def semantic_anchor_match(anchor: Any, evidence: Any) -> bool:
    anchor_tokens = [tok for tok in tokenize(str(anchor or "")) if len(tok) >= 3]
    if not anchor_tokens:
        return True
    body_tokens = set(tokenize(str(evidence or "")))
    return bool(set(anchor_tokens) & body_tokens)


def contradiction_flag(predicted_answer: Any, gold_answer: Any, evidence: Any, question: Any = "") -> bool:
    if not clean_text(predicted_answer) or not clean_text(gold_answer):
        return False
    pred_in_evidence = answer_appears_in_text(predicted_answer, evidence, question)
    gold_in_evidence = answer_appears_in_text(gold_answer, evidence, question)
    pred_correct = answer_equivalence(predicted_answer, gold_answer, question)["answer_correct"]
    return bool(pred_in_evidence and not gold_in_evidence and not pred_correct)


def verify_supported_answer(
    question: Any,
    gold_answer: Any,
    predicted_answer: Any,
    retrieved_evidence: Any = "",
    selected_evidence: Any = "",
    semantic_anchor: Any = "",
    question_type: Any = "",
    mode: str = "hybrid",
) -> dict[str, Any]:
    evidence = clean_text(selected_evidence) or clean_text(retrieved_evidence)
    answer_match = answer_equivalence(predicted_answer, gold_answer, question)
    evidence_supports_gold = answer_appears_in_text(gold_answer, evidence, question)
    evidence_supports_prediction = answer_appears_in_text(predicted_answer, evidence, question)
    anchor_ok = semantic_anchor_match(semantic_anchor, evidence)
    contradiction = contradiction_flag(predicted_answer, gold_answer, evidence, question)
    prediction_supported = bool(evidence_supports_prediction and not contradiction and answer_match["answer_type_valid"])
    supported_answer = bool(answer_match["answer_correct"] and prediction_supported)
    support_score = (
        0.40 * float(evidence_supports_prediction)
        + 0.30 * float(supported_answer)
        + 0.20 * float(answer_match["answer_f1"])
        + 0.10 * float(answer_match["answer_type_valid"])
    )
    if anchor_ok:
        support_score = min(1.0, support_score + 0.05)
    if contradiction:
        support_score = max(0.0, support_score - 0.35)
    if mode == "hybrid_llm":
        reason = "hybrid_llm_requested_but_no_local_llm_configured_fallback_to_deterministic"
    else:
        reason = "deterministic_type_aware_answer_evidence_match"
    if not evidence:
        failure = "missing_evidence"
    elif contradiction:
        failure = "contradiction"
    elif not answer_match["answer_type_valid"]:
        failure = "format_mismatch"
    elif not answer_match["answer_correct"]:
        failure = "wrong_answer"
    elif answer_match["answer_correct"] and not prediction_supported:
        failure = "unsupported_answer"
    elif 0.35 <= support_score < 0.60:
        failure = "verifier_uncertain"
    else:
        failure = "none"
    return {
        "answer_correct": bool(answer_match["answer_correct"]),
        "answer_f1": float(answer_match["answer_f1"]),
        "answer_em": float(answer_match["answer_em"]),
        "answer_type": answer_match["answer_type"],
        "answer_type_valid": bool(answer_match["answer_type_valid"]),
        "evidence_supports_gold": bool(evidence_supports_gold),
        "evidence_supports_prediction": bool(evidence_supports_prediction),
        "prediction_supported_by_evidence": bool(prediction_supported),
        "supported_answer_v2": bool(supported_answer),
        "support_score_v2": float(max(0.0, min(1.0, support_score))),
        "semantic_anchor_match": bool(anchor_ok),
        "contradiction_flag": bool(contradiction),
        "failure_type": failure,
        "reason": reason,
    }


def load_rl_map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in load_jsonl(path)}


def verify_split(split: str, scored_dir: Path, rl_data_dir: Path, out_dir: Path, mode: str) -> list[dict[str, Any]]:
    scored_rows = load_jsonl(scored_dir / f"scored_rollouts_{split}.jsonl")
    rl_rows = load_rl_map(rl_data_dir / f"grounded_rl_{split}.jsonl")
    out = []
    for row in scored_rows:
        sample_id = str(row.get("sample_id") or "")
        example = rl_rows.get(sample_id, {})
        segments = row.get("prediction_segments")
        if not isinstance(segments, dict):
            segments = extract_segments(str(row.get("prediction") or ""), TAGS)
        result = verify_supported_answer(
            question=example.get("question"),
            gold_answer=example.get("gold_answer") or example.get("answer"),
            predicted_answer=segments.get("answer", ""),
            retrieved_evidence=example.get("teacher_evidence_text") or example.get("evidence"),
            selected_evidence=segments.get("evidence", ""),
            semantic_anchor=example.get("semantic_anchor") or example.get("visual_anchor"),
            question_type=example.get("question_type"),
            mode=mode,
        )
        out.append(
            {
                "sample_id": sample_id,
                "split": split,
                "rollout_index": row.get("rollout_index", 0),
                "question": example.get("question"),
                "gold_answer": example.get("gold_answer") or example.get("answer"),
                "predicted_answer": segments.get("answer", ""),
                "selected_evidence": segments.get("evidence", ""),
                "retrieved_evidence": example.get("teacher_evidence_text") or example.get("evidence"),
                "semantic_anchor": example.get("semantic_anchor") or example.get("visual_anchor"),
                **result,
            }
        )
    write_jsonl(out_dir / f"supported_answer_v2_{split}.jsonl", out)
    return out


def summarize(rows_by_split: dict[str, list[dict[str, Any]]]) -> str:
    rows = []
    for split, rows_split in rows_by_split.items():
        n = len(rows_split)
        rows.append(
            {
                "split": split,
                "n": n,
                "answer_correct": mean([float(r["answer_correct"]) for r in rows_split]) if n else 0.0,
                "evidence_supports_prediction": mean([float(r["evidence_supports_prediction"]) for r in rows_split]) if n else 0.0,
                "evidence_supports_gold": mean([float(r["evidence_supports_gold"]) for r in rows_split]) if n else 0.0,
                "supported_answer_v2": mean([float(r["supported_answer_v2"]) for r in rows_split]) if n else 0.0,
                "support_score_v2": mean([float(r["support_score_v2"]) for r in rows_split]) if n else 0.0,
            }
        )
    return "# Supported Answer Verifier v2 Summary\n\n" + md_table(rows) + "\n"


def main() -> int:
    args = parse_args()
    scored_dir = Path(args.scored_dir)
    rl_data_dir = Path(args.rl_data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = {}
    for split in args.splits:
        rows_by_split[split] = verify_split(split, scored_dir, rl_data_dir, out_dir, args.mode)
    (out_dir / "supported_answer_v2_summary.md").write_text(summarize(rows_by_split), encoding="utf-8")
    print(json.dumps({split: len(rows) for split, rows in rows_by_split.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
