#!/usr/bin/env python3
"""Compare direct and forced-search MMSearch-R1 validation outputs.

This diagnostic answers a paper-critical question: which examples actually
benefit from search? DAG-IG should improve the search_helpful subset without
turning search_unnecessary examples into over-search.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.summarize_mmsearch_val_result import (
    as_response_list,
    exact_match,
    extract_answer,
    has_image_search,
    has_text_search,
    normalize_text,
    row_answers,
    substring_match,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", required=True, help="Direct/normal prompt val_result_*.json.")
    parser.add_argument("--search", required=True, help="Forced-search val_result_*.json.")
    parser.add_argument("--method", default="mmsearch_search_need", help="Method label for summary output.")
    parser.add_argument("--correct-threshold", type=float, default=0.1001, help="MMSearch-R1 score threshold.")
    parser.add_argument("--output-csv", default="paper_artifacts/tables/search_need_diagnostic.csv")
    parser.add_argument("--output-json", default="paper_artifacts/tables/search_need_diagnostic.json")
    parser.add_argument("--output-samples-csv", default="paper_artifacts/tables/search_need_samples.csv")
    return parser.parse_args()


def load_rows(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise TypeError(f"Expected JSON list in {path}, got {type(rows)}")
    return [row for row in rows if isinstance(row, dict)]


def row_key(row: dict[str, Any], fallback_index: int) -> str:
    reward_model = row.get("reward_model") or {}
    if not isinstance(reward_model, dict):
        reward_model = {}
    image_url = normalize_text(row.get("image_url"))
    input_text = normalize_text(row.get("input_text"))
    ground_truth = normalize_text(reward_model.get("ground_truth"))
    if image_url or input_text:
        return f"{image_url}|{input_text}|{ground_truth}"
    return f"idx:{fallback_index}"


def row_score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def row_info(row: dict[str, Any], threshold: float) -> dict[str, Any]:
    responses = as_response_list(row.get("output_text"))
    answers = row_answers(row)
    prediction = extract_answer(responses)
    score = row_score(row)
    return {
        "score": score,
        "score_correct": score >= threshold,
        "exact_correct": exact_match(prediction, answers),
        "substring_correct": substring_match(prediction, answers),
        "prediction": prediction,
        "responses": responses,
        "answers": answers,
        "has_image_search": has_image_search(responses),
        "has_text_search": has_text_search(responses),
    }


def classify(direct: dict[str, Any], search: dict[str, Any]) -> str:
    direct_correct = bool(direct["score_correct"])
    search_correct = bool(search["score_correct"])
    search_called = bool(search["has_image_search"] or search["has_text_search"])
    if not search_called:
        return "search_protocol_failed"
    if not direct_correct and search_correct:
        return "search_helpful"
    if direct_correct and search_correct:
        return "search_unnecessary"
    if direct_correct and not search_correct:
        return "search_harmful"
    return "hard"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_diagnostic(
    direct_rows: list[dict[str, Any]],
    search_rows: list[dict[str, Any]],
    threshold: float,
    method: str,
    direct_path: str,
    search_path: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    direct_by_key = {row_key(row, i): row for i, row in enumerate(direct_rows)}
    search_by_key = {row_key(row, i): row for i, row in enumerate(search_rows)}
    keys = sorted(set(direct_by_key) & set(search_by_key))
    sample_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for key in keys:
        direct = row_info(direct_by_key[key], threshold)
        search = row_info(search_by_key[key], threshold)
        group = classify(direct, search)
        counts[group] = counts.get(group, 0) + 1
        sample_rows.append(
            {
                "sample_key": key,
                "group": group,
                "direct_score": direct["score"],
                "search_score": search["score"],
                "score_delta": search["score"] - direct["score"],
                "direct_correct": direct["score_correct"],
                "search_correct": search["score_correct"],
                "direct_substring_correct": direct["substring_correct"],
                "search_substring_correct": search["substring_correct"],
                "search_has_image": search["has_image_search"],
                "search_has_text": search["has_text_search"],
                "direct_prediction": direct["prediction"],
                "search_prediction": search["prediction"],
                "gold_answers": " | ".join(search["answers"] or direct["answers"]),
                "direct_response": " || ".join(direct["responses"]),
                "search_response": " || ".join(search["responses"]),
            }
        )

    n = len(sample_rows)
    search_called = [r for r in sample_rows if r["search_has_image"] or r["search_has_text"]]
    direct_scores = [float(r["direct_score"]) for r in sample_rows]
    search_scores = [float(r["search_score"]) for r in sample_rows]
    summary = {
        "method": method,
        "direct_path": direct_path,
        "search_path": search_path,
        "n_direct": len(direct_rows),
        "n_search": len(search_rows),
        "n_aligned": n,
        "direct_answer_score": mean([1.0 if r["direct_correct"] else 0.0 for r in sample_rows]),
        "search_answer_score": mean([1.0 if r["search_correct"] else 0.0 for r in sample_rows]),
        "score_delta_mean": mean([float(r["score_delta"]) for r in sample_rows]),
        "direct_score_mean": mean(direct_scores),
        "search_score_mean": mean(search_scores),
        "search_call_rate": len(search_called) / n if n else 0.0,
        "image_search_rate": mean([1.0 if r["search_has_image"] else 0.0 for r in sample_rows]),
        "text_search_rate": mean([1.0 if r["search_has_text"] else 0.0 for r in sample_rows]),
        "search_helpful_n": counts.get("search_helpful", 0),
        "search_helpful_rate": counts.get("search_helpful", 0) / n if n else 0.0,
        "search_unnecessary_n": counts.get("search_unnecessary", 0),
        "search_unnecessary_rate": counts.get("search_unnecessary", 0) / n if n else 0.0,
        "search_harmful_n": counts.get("search_harmful", 0),
        "search_harmful_rate": counts.get("search_harmful", 0) / n if n else 0.0,
        "hard_n": counts.get("hard", 0),
        "hard_rate": counts.get("hard", 0) / n if n else 0.0,
        "search_protocol_failed_n": counts.get("search_protocol_failed", 0),
        "search_protocol_failed_rate": counts.get("search_protocol_failed", 0) / n if n else 0.0,
    }
    return summary, sample_rows


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    direct_rows = load_rows(args.direct)
    search_rows = load_rows(args.search)
    summary, sample_rows = build_diagnostic(
        direct_rows=direct_rows,
        search_rows=search_rows,
        threshold=args.correct_threshold,
        method=args.method,
        direct_path=args.direct,
        search_path=args.search,
    )

    for key, value in summary.items():
        print(f"{key}={value}")

    write_csv(args.output_csv, [summary])
    print(f"wrote_csv={args.output_csv}")
    write_csv(args.output_samples_csv, sample_rows)
    print(f"wrote_samples_csv={args.output_samples_csv}")
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": sample_rows}, f, ensure_ascii=False, indent=2)
    print(f"wrote_json={args.output_json}")


if __name__ == "__main__":
    main()
