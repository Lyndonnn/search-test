from __future__ import annotations

import argparse
import ast
import itertools
import json
from typing import Any

from data.dataset_mixer import build_indexes_from_samples
from data.schema import VQASample, sample_to_dict
from utils.io import write_jsonl


DATASET_ALIASES = {
    "fvqa": "lmms-lab/FVQA",
    "infoseek": "HuggingFaceM4/InfoSeek",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a small VQA JSONL for DAG-IG reference reward diagnostics.")
    parser.add_argument("--dataset", default="fvqa", help="Alias or HF dataset name.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--out", default="")
    parser.add_argument("--text-index", default="")
    parser.add_argument("--image-index", default="")
    parser.add_argument("--fallback-to-toy", action="store_true")
    parser.add_argument("--no-streaming", action="store_true")
    return parser.parse_args()


def load_hf_vqa_samples(dataset: str, split: str, limit: int, streaming: bool = True) -> list[VQASample]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("Install datasets to load HF VQA data: pip install datasets") from exc

    dataset_name = DATASET_ALIASES.get(dataset, dataset)
    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    samples: list[VQASample] = []
    for idx, row in enumerate(itertools.islice(ds, limit * 3)):
        sample = row_to_sample(row, idx, dataset)
        if sample.question and sample.gold_answers:
            samples.append(sample)
        if len(samples) >= limit:
            break
    if not samples:
        raise RuntimeError(f"No usable samples found in {dataset_name}:{split}")
    return samples


def row_to_sample(row: dict[str, Any], idx: int, dataset: str) -> VQASample:
    question = _first_text(row, ["question", "query", "prompt", "text", "Question"])
    answers = _answers(row)
    images = _images(row, idx)
    sample_id = str(
        row.get("sample_id")
        or row.get("question_id")
        or row.get("qid")
        or row.get("id")
        or row.get("data_id")
        or row.get("image_id")
        or f"{dataset}_{idx}"
    )
    category = str(row.get("category", "")).lower()
    metadata = {
        "source_dataset": dataset,
        "needs_search": category != "no_search",
        "row_keys": sorted(str(key) for key in row.keys()),
    }
    if row.get("category") is not None:
        metadata["category"] = str(row.get("category"))
    evidence = _first_text(row, ["evidence", "caption", "context", "passage", "wiki_title"])
    if evidence:
        metadata["evidence"] = evidence
    return VQASample(
        sample_id=sample_id,
        question=question,
        images=images,
        gold_answers=answers,
        metadata=metadata,
    )


def _first_text(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            value = row[key]
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, (int, float)):
                return str(value)
            if isinstance(value, dict):
                for nested_key in ("text", "value", "answer", "title"):
                    if nested_key in value and value[nested_key] is not None:
                        return str(value[nested_key]).strip()
            if isinstance(value, (list, tuple)) and value:
                first = value[0]
                if isinstance(first, str):
                    return first.strip()
                if isinstance(first, dict):
                    for nested_key in ("content", "text", "value", "answer", "title"):
                        if nested_key in first and first[nested_key] is not None:
                            return str(first[nested_key]).strip()
                return str(first).strip()
    return ""


def _answers(row: dict[str, Any]) -> list[str]:
    if isinstance(row.get("reward_model"), dict):
        reward_model = row["reward_model"]
        values = []
        ground_truth = reward_model.get("ground_truth")
        if ground_truth is not None and str(ground_truth).strip():
            values.append(str(ground_truth).strip())
        candidates = reward_model.get("candidate_answers")
        values.extend(_parse_answer_list(candidates))
        if values:
            return _dedupe(values)
    for key in ("answers", "answer", "gold_answers", "label", "target", "ground_truth", "candidate_answers"):
        if key not in row or row[key] is None:
            continue
        value = row[key]
        if isinstance(value, str):
            parsed = _parse_answer_list(value)
            return parsed or ([value.strip()] if value.strip() else [])
        if isinstance(value, dict):
            values = []
            for nested_key in ("text", "answer", "value", "label"):
                nested = value.get(nested_key)
                if isinstance(nested, str) and nested.strip():
                    values.append(nested.strip())
                elif isinstance(nested, list):
                    values.extend(str(item).strip() for item in nested if str(item).strip())
            if values:
                return values
        if isinstance(value, (list, tuple)):
            values = []
            for item in value:
                if isinstance(item, dict):
                    text = _first_text(item, ["text", "answer", "value", "label"])
                    if text:
                        values.append(text)
                elif str(item).strip():
                    values.append(str(item).strip())
            if values:
                return _dedupe(values)
    return []


def _images(row: dict[str, Any], idx: int) -> list[str]:
    data_id = row.get("data_id") or row.get("sample_id") or row.get("id") or idx
    for key in ("image_url", "image_urls", "image_path", "image", "images"):
        if key not in row or row[key] is None:
            continue
        value = row[key]
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            result = []
            for item_idx, item in enumerate(value[:3]):
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, dict):
                    if "url" in item:
                        result.append(str(item["url"]))
                    elif "path" in item:
                        result.append(str(item["path"]))
                    else:
                        result.append(f"image://{data_id}_{item_idx}")
                else:
                    result.append(f"image://{data_id}_{item_idx}")
            return result
        if isinstance(value, dict):
            for nested_key in ("path", "url", "id"):
                if nested_key in value and value[nested_key] is not None:
                    return [str(value[nested_key])]
            return [f"image://{data_id}"]
        return [f"image://{data_id}"]
    return [f"image://{data_id}"]


def _parse_answer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return _dedupe([str(item).strip() for item in value if str(item).strip()])
    if not isinstance(value, str):
        return [str(value).strip()] if str(value).strip() else []
    text = value.strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, (list, tuple)):
            return _dedupe([str(item).strip() for item in parsed if str(item).strip()])
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
    return [text]


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        key = value.lower()
        if value and key not in seen:
            out.append(value)
            seen.add(key)
    return out


def main() -> None:
    args = parse_args()
    out = args.out or f"data/processed/{args.dataset}_{args.split}_small.jsonl"
    text_index = args.text_index or f"data/indexes/{args.dataset}_{args.split}_text_corpus.jsonl"
    image_index = args.image_index or f"data/indexes/{args.dataset}_{args.split}_image_corpus.jsonl"
    try:
        samples = load_hf_vqa_samples(args.dataset, args.split, args.limit, streaming=not args.no_streaming)
    except Exception:
        if not args.fallback_to_toy:
            raise
        from data.schema import toy_samples

        samples = toy_samples()[: args.limit]
    write_jsonl(out, [sample_to_dict(sample) for sample in samples])
    build_indexes_from_samples(samples, text_path=text_index, image_path=image_index)
    print(f"wrote samples={out} n={len(samples)}")
    print(f"wrote text_index={text_index}")
    print(f"wrote image_index={image_index}")


if __name__ == "__main__":
    main()
