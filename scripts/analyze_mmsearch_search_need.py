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
    parser.add_argument(
        "--correctness-mode",
        choices=["semantic", "score"],
        default="semantic",
        help=(
            "semantic uses exact/substring match on the answer span, falling back to full response text. "
            "score uses MMSearch-R1's score threshold and is format-sensitive."
        ),
    )
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


def has_image_search_attempt(responses: list[str]) -> bool:
    if has_image_search(responses):
        return True
    for response in responses:
        compact = "".join(str(response).lower().split())
        if "search" in compact and "img" in compact:
            return True
    return False


def has_text_search_attempt(responses: list[str]) -> bool:
    if has_text_search(responses):
        return True
    return any("text_search" in str(response).lower() for response in responses)


def row_info(row: dict[str, Any], threshold: float) -> dict[str, Any]:
    responses = as_response_list(row.get("output_text"))
    answers = row_answers(row)
    prediction = extract_answer(responses)
    response_text = " ".join(responses)
    score = row_score(row)
    answer_exact = exact_match(prediction, answers)
    answer_substring = substring_match(prediction, answers)
    fallback_prediction = prediction or response_text
    semantic_exact = exact_match(fallback_prediction, answers)
    semantic_substring = substring_match(fallback_prediction, answers)
    valid_image_search = has_image_search(responses)
    valid_text_search = has_text_search(responses)
    image_attempt = has_image_search_attempt(responses)
    text_attempt = has_text_search_attempt(responses)
    return {
        "score": score,
        "score_correct": score >= threshold,
        "answer_exact_correct": answer_exact,
        "answer_substring_correct": answer_substring,
        "semantic_exact_correct": semantic_exact,
        "semantic_substring_correct": semantic_substring,
        "semantic_correct": semantic_exact or semantic_substring,
        "prediction": prediction,
        "fallback_prediction": fallback_prediction,
        "responses": responses,
        "answers": answers,
        "has_image_search": valid_image_search,
        "has_text_search": valid_text_search,
        "has_image_search_attempt": image_attempt,
        "has_text_search_attempt": text_attempt,
        "has_malformed_search_attempt": (image_attempt and not valid_image_search)
        or (text_attempt and not valid_text_search),
    }


def is_correct(info: dict[str, Any], correctness_mode: str) -> bool:
    if correctness_mode == "score":
        return bool(info["score_correct"])
    return bool(info["semantic_correct"])


def classify(direct: dict[str, Any], search: dict[str, Any], correctness_mode: str) -> str:
    direct_correct = is_correct(direct, correctness_mode)
    search_correct = is_correct(search, correctness_mode)
    search_called = bool(search["has_image_search"] or search["has_text_search"])
    if not search_called:
        if search["has_image_search_attempt"] or search["has_text_search_attempt"]:
            return "malformed_search_attempt"
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
    correctness_mode: str = "semantic",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    direct_by_key = {row_key(row, i): row for i, row in enumerate(direct_rows)}
    search_by_key = {row_key(row, i): row for i, row in enumerate(search_rows)}
    keys = sorted(set(direct_by_key) & set(search_by_key))
    sample_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for key in keys:
        direct = row_info(direct_by_key[key], threshold)
        search = row_info(search_by_key[key], threshold)
        group = classify(direct, search, correctness_mode)
        counts[group] = counts.get(group, 0) + 1
        sample_rows.append(
            {
                "sample_key": key,
                "group": group,
                "direct_score": direct["score"],
                "search_score": search["score"],
                "score_delta": search["score"] - direct["score"],
                "direct_correct": is_correct(direct, correctness_mode),
                "search_correct": is_correct(search, correctness_mode),
                "direct_score_correct": direct["score_correct"],
                "search_score_correct": search["score_correct"],
                "direct_answer_substring_correct": direct["answer_substring_correct"],
                "search_answer_substring_correct": search["answer_substring_correct"],
                "direct_semantic_correct": direct["semantic_correct"],
                "search_semantic_correct": search["semantic_correct"],
                "search_has_image": search["has_image_search"],
                "search_has_text": search["has_text_search"],
                "search_attempted_image": search["has_image_search_attempt"],
                "search_attempted_text": search["has_text_search_attempt"],
                "search_malformed_attempt": search["has_malformed_search_attempt"],
                "direct_prediction": direct["prediction"],
                "search_prediction": search["prediction"],
                "direct_fallback_prediction": direct["fallback_prediction"],
                "search_fallback_prediction": search["fallback_prediction"],
                "gold_answers": " | ".join(search["answers"] or direct["answers"]),
                "direct_response": " || ".join(direct["responses"]),
                "search_response": " || ".join(search["responses"]),
            }
        )

    n = len(sample_rows)
    search_called = [r for r in sample_rows if r["search_has_image"] or r["search_has_text"]]
    search_attempted = [r for r in sample_rows if r["search_attempted_image"] or r["search_attempted_text"]]
    malformed_attempts = [r for r in sample_rows if r["search_malformed_attempt"]]
    direct_scores = [float(r["direct_score"]) for r in sample_rows]
    search_scores = [float(r["search_score"]) for r in sample_rows]
    summary = {
        "method": method,
        "direct_path": direct_path,
        "search_path": search_path,
        "correctness_mode": correctness_mode,
        "n_direct": len(direct_rows),
        "n_search": len(search_rows),
        "n_aligned": n,
        "direct_answer_score": mean([1.0 if r["direct_correct"] else 0.0 for r in sample_rows]),
        "search_answer_score": mean([1.0 if r["search_correct"] else 0.0 for r in sample_rows]),
        "direct_score_correct_rate": mean([1.0 if r["direct_score_correct"] else 0.0 for r in sample_rows]),
        "search_score_correct_rate": mean([1.0 if r["search_score_correct"] else 0.0 for r in sample_rows]),
        "direct_semantic_correct_rate": mean([1.0 if r["direct_semantic_correct"] else 0.0 for r in sample_rows]),
        "search_semantic_correct_rate": mean([1.0 if r["search_semantic_correct"] else 0.0 for r in sample_rows]),
        "score_delta_mean": mean([float(r["score_delta"]) for r in sample_rows]),
        "direct_score_mean": mean(direct_scores),
        "search_score_mean": mean(search_scores),
        "search_call_rate": len(search_called) / n if n else 0.0,
        "search_attempt_rate": len(search_attempted) / n if n else 0.0,
        "malformed_search_attempt_rate": len(malformed_attempts) / n if n else 0.0,
        "image_search_rate": mean([1.0 if r["search_has_image"] else 0.0 for r in sample_rows]),
        "image_search_attempt_rate": mean([1.0 if r["search_attempted_image"] else 0.0 for r in sample_rows]),
        "text_search_rate": mean([1.0 if r["search_has_text"] else 0.0 for r in sample_rows]),
        "text_search_attempt_rate": mean([1.0 if r["search_attempted_text"] else 0.0 for r in sample_rows]),
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
        "malformed_search_attempt_n": counts.get("malformed_search_attempt", 0),
        "malformed_search_attempt_group_rate": counts.get("malformed_search_attempt", 0) / n if n else 0.0,
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
        correctness_mode=args.correctness_mode,
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
