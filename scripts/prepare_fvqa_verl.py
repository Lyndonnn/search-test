#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import re
from io import BytesIO
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert FVQA to a small veRL-format parquet for MMSearch-R1.")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="FVQA split to convert.")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to export.")
    parser.add_argument("--offset", type=int, default=0, help="Start offset inside the split.")
    parser.add_argument(
        "--out",
        required=True,
        help="Output parquet path, e.g. mmsearch_r1/data/fvqa_debug_train.pq",
    )
    parser.add_argument(
        "--data-source",
        default="mmsearch_r1/fvqa",
        help="data_source value written into the parquet; must contain 'mmsearch_r1' for current reward dispatch.",
    )
    parser.add_argument(
        "--image-key",
        default="images",
        help="Column name for images in the output parquet.",
    )
    parser.add_argument(
        "--print-sample",
        action="store_true",
        help="Print the first converted sample for inspection.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "Read FVQA with HuggingFace streaming and then apply offset/limit locally. "
            "This avoids downloading the full FVQA parquet for small debug runs."
        ),
    )
    parser.add_argument(
        "--image-dir",
        default="",
        help="Optional directory to export images and write their local paths into image_urls.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if isinstance(value, list):
        parts = [normalize_text(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("content", "text", "question", "prompt", "query", "value"):
            if key in value:
                text = normalize_text(value[key])
                if text:
                    return text
        parts = [normalize_text(v) for v in value.values()]
        return " ".join(part for part in parts if part).strip()
    return str(value).strip()


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text == "[]":
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return normalize_string_list(parsed)
        return [text]
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            text = normalize_text(item)
            if text:
                items.append(text)
        return items
    if isinstance(value, dict):
        text = normalize_text(value)
        return [text] if text else []
    text = normalize_text(value)
    return [text] if text else []


def choose_question(example: dict[str, Any]) -> str:
    for key in ("question", "query", "prompt", "messages"):
        text = normalize_text(example.get(key))
        if text:
            return text
    raise ValueError(f"Unable to find question text in sample keys: {list(example.keys())}")


def choose_answers(example: dict[str, Any]) -> tuple[str, list[str]]:
    reward_model = example.get("reward_model")
    if isinstance(reward_model, dict):
        candidates = normalize_string_list(reward_model.get("ground_truth"))
        candidates += normalize_string_list(reward_model.get("candidate_answers"))
        if candidates:
            ground_truth = candidates[0]
            alt_answers = []
            seen = {ground_truth}
            for ans in candidates[1:]:
                if ans not in seen:
                    alt_answers.append(ans)
                    seen.add(ans)
            return ground_truth, alt_answers

    candidates: list[str] = []
    for key in ("answer", "answers", "label", "labels", "ground_truth"):
        candidates = normalize_string_list(example.get(key))
        if candidates:
            break
    if not candidates:
        raise ValueError(f"Unable to find answer text in sample keys: {list(example.keys())}")
    ground_truth = candidates[0]
    alt_answers = []
    seen = {ground_truth}
    for ans in candidates[1:]:
        if ans not in seen:
            alt_answers.append(ans)
            seen.add(ans)
    return ground_truth, alt_answers


def image_to_bytes(value: Any) -> bytes:
    if value is None:
        raise ValueError("FVQA sample does not contain an image.")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return value["bytes"]
        if value.get("path"):
            with open(value["path"], "rb") as f:
                return f.read()
    if isinstance(value, bytes):
        return value
    if isinstance(value, Image.Image):
        image = value
    else:
        raise TypeError(f"Unsupported image type: {type(value)}")
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def choose_image(example: dict[str, Any]) -> bytes:
    existing_images = example.get("images")
    if isinstance(existing_images, list) and existing_images:
        first_image = existing_images[0]
        if first_image is not None:
            return image_to_bytes(first_image)

    for key in ("image", "img"):
        if key in example and example[key] is not None:
            return image_to_bytes(example[key])
    raise ValueError(f"Unable to find image in sample keys: {list(example.keys())}")


def choose_image_url(example: dict[str, Any]) -> str:
    for key in ("image_urls", "image_url", "url"):
        text = normalize_text(example.get(key))
        if text:
            return text
    return ""


def choose_id(example: dict[str, Any], fallback: int) -> str:
    for key in ("data_id", "id", "qid", "question_id", "uid"):
        text = normalize_text(example.get(key))
        if text:
            return text
    return str(fallback)


def safe_filename(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return normalized or "sample"


def export_search_image(image_bytes: bytes, image_dir: str, split: str, qid: str) -> str:
    os.makedirs(image_dir, exist_ok=True)
    path = os.path.join(image_dir, f"{safe_filename(split)}_{safe_filename(qid)}.png")
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.save(path)
    return path


def normalize_prompt(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, list):
        normalized = []
        for turn in prompt:
            if isinstance(turn, dict):
                role = normalize_text(turn.get("role")) or "user"
                content = normalize_text(turn.get("content"))
                normalized.append({"role": role, "content": content})
            else:
                content = normalize_text(turn)
                if content:
                    normalized.append({"role": "user", "content": content})
        if normalized:
            return normalized

    text = normalize_text(prompt)
    if text:
        return [{"role": "user", "content": text}]
    raise ValueError("Unable to normalize prompt into chat format.")


def normalize_images_field(value: Any) -> list[dict[str, bytes]]:
    if isinstance(value, list) and value:
        normalized = []
        for image in value:
            normalized.append({"bytes": image_to_bytes(image)})
        return normalized

    if value is not None:
        return [{"bytes": image_to_bytes(value)}]

    return []


def normalize_reward_model_field(value: Any, ground_truth: str, candidate_answers: list[str]) -> dict[str, Any]:
    reward_model = value if isinstance(value, dict) else {}
    style = normalize_text(reward_model.get("style")) or "rule"
    out = {
        "style": style,
        "ground_truth": ground_truth,
        # Keep the original MMSearch-R1 schema: candidate_answers is a JSON
        # string, not a nested Arrow list that can reload as numpy.ndarray.
        "candidate_answers": json.dumps(candidate_answers, ensure_ascii=False),
    }
    return out


def build_record(
    example: dict[str, Any],
    idx: int,
    split: str,
    data_source: str,
    image_key: str,
    image_dir: str = "",
) -> dict[str, Any]:
    ground_truth, candidate_answers = choose_answers(example)
    qid = choose_id(example, idx)

    if "prompt" in example and example.get("prompt") is not None:
        prompt = normalize_prompt(example.get("prompt"))
    else:
        question = choose_question(example)
        prompt = [{"role": "user", "content": question}]

    images = normalize_images_field(example.get(image_key))
    if not images:
        images = [{"bytes": choose_image(example)}]

    record_data_source = normalize_text(example.get("data_source")) or data_source
    image_url = choose_image_url(example)
    if not image_url and image_dir:
        image_url = export_search_image(images[0]["bytes"], image_dir=image_dir, split=split, qid=qid)

    record = {
        "prompt": prompt,
        image_key: images,
        "reward_model": normalize_reward_model_field(example.get("reward_model"), ground_truth, candidate_answers),
        "data_source": record_data_source,
        "image_urls": image_url,
        "extra_info": {
            "index": idx,
            "question_id": qid,
            "source_split": split,
        },
    }
    category = normalize_text(example.get("category"))
    if category:
        record["extra_info"]["category"] = category
    return record


def main() -> None:
    args = parse_args()
    if args.streaming:
        dataset = load_dataset("lmms-lab/FVQA", split=args.split, streaming=True)
        dataset = itertools.islice(dataset, args.offset, args.offset + args.limit)
    else:
        hf_split = f"{args.split}[{args.offset}:{args.offset + args.limit}]"
        dataset = load_dataset("lmms-lab/FVQA", split=hf_split)

    records = [
        build_record(
            example,
            idx=args.offset + i,
            split=args.split,
            data_source=args.data_source,
            image_key=args.image_key,
            image_dir=args.image_dir,
        )
        for i, example in enumerate(dataset)
    ]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, args.out)

    print(f"Saved {len(records)} records to {args.out}")
    if args.print_sample and records:
        sample = dict(records[0])
        sample[args.image_key] = [{"bytes": f"<{len(records[0][args.image_key][0]['bytes'])} bytes>"}]
        print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
