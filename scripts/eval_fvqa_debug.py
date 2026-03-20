#!/usr/bin/env python3
import argparse
import json
import os
from io import BytesIO
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
from PIL import Image

from mmsearch_r1.scripts.inference_torch_demo import load_model_and_processor, run_mmsearch_demo
from mmsearch_r1.utils.reward_score_mm.mmsearch_r1_score import (
    compute_score,
    em_check,
    extract_solution,
    subem_check,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMSearch-R1 torch-demo baseline evaluation on FVQA parquet.")
    parser.add_argument("--parquet", required=True, help="Path to veRL-format FVQA parquet.")
    parser.add_argument("--model-path", required=True, help="HF model path, e.g. lmms-lab/MMSearch-R1-7B")
    parser.add_argument("--output", required=True, help="Path to save per-sample JSON results.")
    parser.add_argument("--limit", type=int, default=20, help="Number of rows to evaluate.")
    parser.add_argument("--offset", type=int, default=0, help="Start row offset.")
    return parser.parse_args()


def as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    return [value]


def as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "item"):
        maybe = value.item()
        if isinstance(maybe, dict):
            return maybe
    return {}


def export_image(image_payload: dict[str, Any], path: str) -> None:
    image = Image.open(BytesIO(image_payload["bytes"]))
    image.save(path)


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.parquet)
    end = min(len(df), args.offset + args.limit)
    model, processor = load_model_and_processor(args.model_path)

    results = []
    with TemporaryDirectory(prefix="fvqa_eval_") as tmpdir:
        for row_idx in range(args.offset, end):
            row = df.iloc[row_idx]
            prompt = as_sequence(row["prompt"])
            reward_model = as_mapping(row.get("reward_model", {}))
            extra_info = as_mapping(row.get("extra_info", {}))
            images = as_sequence(row["images"])
            if not images:
                raise ValueError(f"Row {row_idx} has no images")

            question = prompt[0].get("content", "") if prompt and isinstance(prompt[0], dict) else str(prompt)
            question_id = extra_info.get("question_id", row_idx)
            image_path = os.path.join(tmpdir, f"{question_id}.png")
            image_payload = images[0]
            if not isinstance(image_payload, dict) and hasattr(image_payload, "item"):
                image_payload = image_payload.item()
            export_image(image_payload, image_path)

            trajectory = run_mmsearch_demo(model, processor, image_path, question)
            ground_truth = reward_model.get("ground_truth", "")
            candidate_answers = reward_model.get("candidate_answers", [])
            if isinstance(candidate_answers, str):
                try:
                    candidate_answers = json.loads(candidate_answers)
                except Exception:
                    candidate_answers = [candidate_answers]
            all_answers = [ground_truth] + [a for a in as_sequence(candidate_answers) if isinstance(a, str)]

            final_answer = extract_solution(trajectory["final_response"])
            em = bool(final_answer is not None and em_check(final_answer, all_answers))
            subem = bool(final_answer is not None and subem_check(final_answer, all_answers))
            score = compute_score(
                prediction=trajectory["responses"],
                ground_truth=all_answers,
                extra_info={"search_penalty": 0.1, "format_penalty": 0.1, "reward_mode": "SubEM"},
            )

            search_count = sum(
                int("<search><img></search>" in resp) + int("<text_search>" in resp and "</text_search>" in resp)
                for resp in trajectory["responses"]
            )

            result = {
                "row_index": row_idx,
                "question_id": question_id,
                "question": question,
                "ground_truth": ground_truth,
                "candidate_answers": candidate_answers,
                "final_answer": final_answer,
                "em": em,
                "subem": subem,
                "score": score,
                "search_count": search_count,
                "responses": trajectory["responses"],
                "tool_trace": trajectory["tool_trace"],
            }
            results.append(result)
            print(
                f"[{ordinal(len(results))}/{end - args.offset}] qid={question_id} em={int(em)} subem={int(subem)} searches={search_count}"
            )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total = len(results)
    em_rate = sum(r["em"] for r in results) / total if total else 0.0
    subem_rate = sum(r["subem"] for r in results) / total if total else 0.0
    avg_searches = sum(r["search_count"] for r in results) / total if total else 0.0
    print(f"Saved results to {args.output}")
    print(f"EM={em_rate:.4f} SubEM={subem_rate:.4f} AvgSearches={avg_searches:.4f}")


if __name__ == "__main__":
    main()
