#!/usr/bin/env python3
"""Inspect grounded RL rows for target leakage before RL."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from grounded_pipeline_utils import extract_segments, load_jsonl, md_table, write_csv, write_jsonl


TAGS = ["ground", "observe", "search_decision", "search", "evidence", "answer"]
WRONG_FIELD_TAGS = ["ground", "observe", "search"]
FORBIDDEN_WORD_RE = re.compile(r"\b(?:bbox|bounding box|red\s*box|red-box|annotation|annotated)\b", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{6,}\d)")
PRICE_RE = re.compile(r"(?:[$€£¥]\s*\d|\b\d+(?:\.\d+)?\s*(?:usd|eur|dollars?|euros?|yuan|yen)\b)", re.I)
DATE_RE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.I,
)
ADDRESS_RE = re.compile(r"\b(?:street|st\.|road|rd\.|avenue|ave\.|boulevard|blvd|suite|floor|zip code)\b", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/dagig_rn03_10_grounded_rl")
    parser.add_argument("--out_dir", default="results/dagig_rn03_10_grounded_rl/leakage")
    return parser.parse_args()


def normalize(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def answer_in_text(answer: Any, text: str) -> bool:
    ans = normalize(answer)
    body = normalize(text)
    if not ans or not body:
        return False
    compact_ans = ans.replace(" ", "")
    min_len = 2 if compact_ans.isdigit() else 3
    return len(compact_ans) >= min_len and ans in body


def typed_leakage(text: str) -> list[str]:
    flags = []
    for name, regex in [
        ("email", EMAIL_RE),
        ("phone", PHONE_RE),
        ("price", PRICE_RE),
        ("date", DATE_RE),
        ("address", ADDRESS_RE),
    ]:
        if regex.search(text or ""):
            flags.append(name)
    return flags


def inspect_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    text = str(row.get("model_target_text") or row.get("original_model_target_text") or "")
    segments = row.get("target_segments") if isinstance(row.get("target_segments"), dict) else extract_segments(text, TAGS)
    answer = row.get("gold_answer") or row.get("answer") or segments.get("answer")
    before_answer = text.split("<answer>", 1)[0] if "<answer>" in text else text
    flags: dict[str, Any] = {}
    for tag in WRONG_FIELD_TAGS:
        flags[f"answer_in_{tag}"] = answer_in_text(answer, str(segments.get(tag, "")))
    flags["answer_before_answer_anywhere"] = answer_in_text(answer, before_answer)
    evidence = str(segments.get("evidence", ""))
    pre_answer_without_evidence = before_answer.replace(evidence, "")
    flags["answer_before_answer_outside_evidence"] = answer_in_text(answer, pre_answer_without_evidence)
    flags["forbidden_wording"] = bool(FORBIDDEN_WORD_RE.search(text))
    typed_flags: list[str] = []
    for tag in WRONG_FIELD_TAGS:
        for hit in typed_leakage(str(segments.get(tag, ""))):
            typed_flags.append(f"{hit}_in_{tag}")
    flags["typed_leakage_wrong_field"] = ",".join(sorted(set(typed_flags)))
    flags["typed_leakage_sensitive"] = False
    sensitive = any(
        bool(flags[key])
        for key in [
            "answer_in_ground",
            "answer_in_observe",
            "answer_in_search",
            "answer_before_answer_outside_evidence",
            "forbidden_wording",
        ]
    )
    row["leakage_sensitive_exclude"] = bool(sensitive)
    row["leakage_flags"] = flags
    report = {
        "sample_id": row.get("sample_id"),
        "split": row.get("split"),
        "leakage_sensitive_exclude": bool(sensitive),
        **flags,
    }
    return row, report


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for split in ["train", "dev", "test"]:
        path = data_dir / f"grounded_rl_{split}.jsonl"
        rows = load_jsonl(path)
        updated = []
        for row in rows:
            new_row, report = inspect_row(row)
            updated.append(new_row)
            reports.append(report)
        write_jsonl(path, updated)
    write_csv(out_dir / "leakage_report.csv", reports)
    write_jsonl(out_dir / "leakage_cases.jsonl", [row for row in reports if row["leakage_sensitive_exclude"]])
    totals: list[dict[str, Any]] = []
    for split in ["train", "dev", "test"]:
        subset = [row for row in reports if row["split"] == split]
        totals.append(
            {
                "split": split,
                "n": len(subset),
                "leakage_sensitive_exclude": sum(1 for row in subset if row["leakage_sensitive_exclude"]),
                "answer_in_ground": sum(1 for row in subset if row["answer_in_ground"]),
                "answer_in_observe": sum(1 for row in subset if row["answer_in_observe"]),
                "answer_in_search": sum(1 for row in subset if row["answer_in_search"]),
                "answer_before_answer_anywhere": sum(1 for row in subset if row["answer_before_answer_anywhere"]),
                "answer_before_answer_outside_evidence": sum(
                    1 for row in subset if row["answer_before_answer_outside_evidence"]
                ),
                "forbidden_wording": sum(1 for row in subset if row["forbidden_wording"]),
                "typed_leakage_wrong_field": sum(1 for row in subset if row["typed_leakage_wrong_field"]),
                "typed_leakage_sensitive": sum(1 for row in subset if row["typed_leakage_sensitive"]),
            }
        )
    md = [
        "# Grounded RL Leakage Inspection",
        "",
        "Rows with `leakage_sensitive_exclude=true` are kept for diagnostics but excluded from RL training reward computation.",
        "",
        md_table(totals),
        "",
        "Note: evidence is allowed to contain answer-bearing support text; `answer_before_answer_anywhere` is reported separately and is not by itself an exclusion trigger.",
        "Typed phone/email/address/price/date patterns are reported for audit, but only answer leakage or forbidden annotation wording triggers exclusion.",
    ]
    (out_dir / "leakage_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
