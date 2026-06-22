#!/usr/bin/env python3
"""Diagnose why DAGIG_v3_total under-predicts supported-answer success."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from statistics import mean
from typing import Any

from grounded_pipeline_utils import extract_segments, load_jsonl, md_table, write_csv


VERIFIER_PATH = Path(__file__).with_name("49_supported_answer_verifier_v2.py")
SPEC = importlib.util.spec_from_file_location("supported_answer_verifier_v2", VERIFIER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import verifier from {VERIFIER_PATH}")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
verify_supported_answer = VERIFIER.verify_supported_answer


SPLITS = ["train", "dev", "test"]
CASE_SPLITS = ["dev", "test"]
TAGS = ["ground", "observe", "search_decision", "search", "evidence", "answer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored_dir", default="results/dagig_rn03_10_counterfactual_dagig_v3")
    parser.add_argument("--rl_data_dir", default="data/dagig_rn03_10_grounded_rl")
    parser.add_argument("--out_dir", default="results/dagig_rn03_10_counterfactual_dagig_v31")
    parser.add_argument("--examples_per_category", type=int, default=30)
    return parser.parse_args()


def load_rl_map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in load_jsonl(path)}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * q)))
    return vals[idx]


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return False


def diagnose_failure(row: dict[str, Any], verifier: dict[str, Any]) -> str:
    if not row.get("prediction_segments", {}).get("answer"):
        return "answer_extracted_incorrectly"
    old_supported = bool_value(row.get("supported_answer"))
    if not old_supported and verifier.get("supported_answer_v2"):
        return "evidence_verifier_too_weak"
    if not verifier.get("answer_type_valid"):
        return "answer_format_mismatch"
    if verifier.get("evidence_supports_gold") and not verifier.get("answer_correct"):
        return "correct_evidence_but_wrong_final_answer"
    if verifier.get("answer_correct") and not verifier.get("evidence_supports_prediction"):
        return "correct_answer_but_unsupported_evidence"
    if bool_value(row.get("retrieval_r5")) and not verifier.get("evidence_supports_prediction"):
        return "query_retrieves_support_but_evidence_selection_bad"
    if not verifier.get("evidence_supports_prediction"):
        return "evidence_not_supporting_answer"
    if not verifier.get("answer_correct"):
        return "answer_string_or_value_mismatch"
    return "other"


def enrich_rows(split: str, scored_dir: Path, rl_data_dir: Path) -> list[dict[str, Any]]:
    scored_rows = load_jsonl(scored_dir / f"scored_rollouts_{split}.jsonl")
    rl_rows = load_rl_map(rl_data_dir / f"grounded_rl_{split}.jsonl")
    enriched = []
    for row in scored_rows:
        sample_id = str(row.get("sample_id") or "")
        example = rl_rows.get(sample_id, {})
        segments = row.get("prediction_segments")
        if not isinstance(segments, dict):
            segments = extract_segments(str(row.get("prediction") or ""), TAGS)
        verifier = verify_supported_answer(
            question=example.get("question"),
            gold_answer=example.get("gold_answer") or example.get("answer"),
            predicted_answer=segments.get("answer", ""),
            retrieved_evidence=example.get("teacher_evidence_text") or example.get("evidence"),
            selected_evidence=segments.get("evidence", ""),
            semantic_anchor=example.get("semantic_anchor") or example.get("visual_anchor"),
            question_type=example.get("question_type"),
        )
        enriched.append(
            {
                **row,
                "prediction_segments": segments,
                "question": example.get("question"),
                "gold_answer": example.get("gold_answer") or example.get("answer"),
                "semantic_anchor": example.get("semantic_anchor") or example.get("visual_anchor"),
                "question_type": example.get("question_type"),
                "verifier_v2": verifier,
                "failure_cause": diagnose_failure({**row, "prediction_segments": segments}, verifier),
            }
        )
    return enriched


def case_row(row: dict[str, Any], category: str) -> dict[str, Any]:
    seg = row.get("prediction_segments") or {}
    v = row.get("verifier_v2") or {}
    return {
        "category": category,
        "sample_id": row.get("sample_id"),
        "rollout_index": row.get("rollout_index"),
        "DAGIG_v3_total": row.get("DAGIG_v3_total"),
        "R_search_v3": row.get("R_search_v3"),
        "R_evidence": row.get("R_evidence"),
        "R_answer_old": row.get("R_answer"),
        "old_supported_answer": row.get("supported_answer"),
        "supported_answer_v2": v.get("supported_answer_v2"),
        "answer_correct_v2": v.get("answer_correct"),
        "answer_f1_v2": v.get("answer_f1"),
        "evidence_support_old": row.get("evidence_support"),
        "evidence_supports_gold_v2": v.get("evidence_supports_gold"),
        "evidence_supports_prediction_v2": v.get("evidence_supports_prediction"),
        "retrieval_r5": row.get("retrieval_r5"),
        "retrieval_mrr": row.get("retrieval_mrr"),
        "failure_cause": row.get("failure_cause"),
        "question": row.get("question"),
        "semantic_anchor": row.get("semantic_anchor"),
        "gold_answer": row.get("gold_answer"),
        "predicted_answer": seg.get("answer"),
        "selected_evidence": seg.get("evidence"),
        "search": seg.get("search"),
    }


def collect_cases(rows: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    totals = [float(row.get("DAGIG_v3_total") or 0.0) for row in rows]
    q75 = percentile(totals, 0.75)
    q25 = percentile(totals, 0.25)
    categories = {
        "high_DAGIG_false_supported_answer": [
            row for row in rows if float(row.get("DAGIG_v3_total") or 0.0) >= q75 and not bool_value(row.get("supported_answer"))
        ],
        "low_DAGIG_true_supported_answer": [
            row for row in rows if float(row.get("DAGIG_v3_total") or 0.0) <= q25 and bool_value(row.get("supported_answer"))
        ],
        "evidence_support_true_but_answer_wrong": [
            row for row in rows if bool_value(row.get("evidence_support")) and not row.get("verifier_v2", {}).get("answer_correct")
        ],
        "answer_correct_but_evidence_support_false": [
            row
            for row in rows
            if row.get("verifier_v2", {}).get("answer_correct")
            and not row.get("verifier_v2", {}).get("evidence_supports_prediction")
        ],
    }
    out = []
    counts = {}
    for name, items in categories.items():
        reverse = name != "low_DAGIG_true_supported_answer"
        items = sorted(items, key=lambda row: float(row.get("DAGIG_v3_total") or 0.0), reverse=reverse)
        counts[name] = len(items)
        out.extend(case_row(row, name) for row in items[:limit])
    return out, {"q25": q25, "q75": q75, **counts}


def aggregate_failure_causes(rows_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for split, rows in rows_by_split.items():
        counts: dict[str, int] = {}
        for row in rows:
            if bool_value(row.get("supported_answer")):
                continue
            cause = str(row.get("failure_cause") or "unknown")
            counts[cause] = counts.get(cause, 0) + 1
        total = sum(counts.values())
        for cause, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
            out.append({"split": split, "failure_cause": cause, "count": count, "rate_among_old_unsupported": count / max(1, total)})
    return out


def main() -> int:
    args = parse_args()
    scored_dir = Path(args.scored_dir)
    rl_data_dir = Path(args.rl_data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = {split: enrich_rows(split, scored_dir, rl_data_dir) for split in SPLITS}
    case_summaries = []
    for split in CASE_SPLITS:
        cases, summary = collect_cases(rows_by_split[split], args.examples_per_category)
        write_csv(out_dir / f"supported_answer_failure_cases_{split}.csv", cases)
        case_summaries.append({"split": split, **summary, "examples_written": len(cases)})
    cause_rows = aggregate_failure_causes(rows_by_split)
    metrics = []
    for split, rows in rows_by_split.items():
        metrics.append(
            {
                "split": split,
                "n": len(rows),
                "old_supported_answer": mean([float(bool_value(r.get("supported_answer"))) for r in rows]) if rows else 0.0,
                "supported_answer_v2": mean([float(r["verifier_v2"]["supported_answer_v2"]) for r in rows]) if rows else 0.0,
                "old_false_v2_true": mean(
                    [float((not bool_value(r.get("supported_answer"))) and r["verifier_v2"]["supported_answer_v2"]) for r in rows]
                )
                if rows
                else 0.0,
                "old_true_v2_false": mean(
                    [float(bool_value(r.get("supported_answer")) and not r["verifier_v2"]["supported_answer_v2"]) for r in rows]
                )
                if rows
                else 0.0,
            }
        )
    md = [
        "# Supported-Answer Failure Diagnosis",
        "",
        "## Old vs v2 Supported-Answer Rates",
        "",
        md_table(metrics),
        "",
        "## Case Sampling Summary",
        "",
        md_table(case_summaries),
        "",
        "## Failure Cause Breakdown",
        "",
        md_table(cause_rows),
    ]
    (out_dir / "supported_answer_failure_diagnosis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"case_summaries": case_summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
