#!/usr/bin/env python3
"""Summarize MMSearch-R1 val-only JSON saved by ray_trainer.py."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Any


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to val_result_*.json.")
    parser.add_argument("--output-csv", default="", help="Optional CSV path for aggregate metrics.")
    parser.add_argument("--output-json", default="", help="Optional JSON path for aggregate metrics.")
    return parser.parse_args()


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def normalize_answer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return normalize_answer_list(parsed)
        return [text]
    if hasattr(value, "tolist"):
        return normalize_answer_list(value.tolist())
    if isinstance(value, dict):
        return []
    if isinstance(value, (list, tuple, set)):
        answers: list[str] = []
        for item in value:
            answers.extend(normalize_answer_list(item))
        return answers
    return [str(value).strip()] if str(value).strip() else []


def as_response_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def extract_answer(responses: list[str]) -> str:
    for response in reversed(responses):
        matches = ANSWER_RE.findall(response)
        if matches:
            return matches[-1].strip()
    return ""


def exact_match(prediction: str, answers: list[str]) -> bool:
    pred = normalize_text(prediction)
    return bool(pred) and any(pred == normalize_text(answer) for answer in answers)


def substring_match(prediction: str, answers: list[str]) -> bool:
    pred = normalize_text(prediction)
    if not pred:
        return False
    for answer in answers:
        gold = normalize_text(answer)
        if gold and (gold in pred or pred in gold):
            return True
    return False


def has_text_search(responses: list[str]) -> bool:
    return any("<text_search>" in response and "</text_search>" in response for response in responses)


def has_image_search(responses: list[str]) -> bool:
    return any("<search><img></search>" in response for response in responses)


def row_answers(row: dict[str, Any]) -> list[str]:
    reward_model = row.get("reward_model") or {}
    if not isinstance(reward_model, dict):
        return []
    answers = normalize_answer_list(reward_model.get("ground_truth"))
    answers.extend(normalize_answer_list(reward_model.get("candidate_answers")))
    seen = set()
    deduped = []
    for answer in answers:
        key = normalize_text(answer)
        if key and key not in seen:
            deduped.append(answer)
            seen.add(key)
    return deduped


def summarize(rows: list[dict[str, Any]], source_path: str) -> dict[str, Any]:
    n = len(rows)
    scores = [float(row.get("score", 0.0) or 0.0) for row in rows]
    em_count = 0
    subem_count = 0
    text_search_count = 0
    image_search_count = 0
    answer_count = 0

    for row in rows:
        responses = as_response_list(row.get("output_text"))
        answers = row_answers(row)
        prediction = extract_answer(responses)
        if prediction:
            answer_count += 1
        if exact_match(prediction, answers):
            em_count += 1
        if substring_match(prediction, answers):
            subem_count += 1
        if has_text_search(responses):
            text_search_count += 1
        if has_image_search(responses):
            image_search_count += 1

    return {
        "source": source_path,
        "n": n,
        "score_mean": sum(scores) / n if n else 0.0,
        "answer_rate": answer_count / n if n else 0.0,
        "answer_em": em_count / n if n else 0.0,
        "answer_subem": subem_count / n if n else 0.0,
        "text_search_rate": text_search_count / n if n else 0.0,
        "image_search_rate": image_search_count / n if n else 0.0,
        "any_search_rate": (text_search_count + image_search_count) / n if n else 0.0,
    }


def write_csv(path: str, metrics: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def main() -> None:
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise TypeError(f"Expected a JSON list in {args.input}, got {type(rows)}")

    metrics = summarize(rows, args.input)
    for key, value in metrics.items():
        print(f"{key}={value}")

    if args.output_csv:
        write_csv(args.output_csv, metrics)
        print(f"wrote_csv={args.output_csv}")
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"wrote_json={args.output_json}")


if __name__ == "__main__":
    main()
