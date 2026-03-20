#!/usr/bin/env python3
import argparse
import os
from io import BytesIO
from typing import Any

import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one veRL parquet sample to an image file and print metadata.")
    parser.add_argument("--parquet", required=True, help="Path to veRL parquet file.")
    parser.add_argument("--index", type=int, default=0, help="Row index to export.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the image.")
    parser.add_argument("--image-key", default="images", help="Image column name.")
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


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.parquet)
    if len(df) == 0:
        raise ValueError(f"Empty parquet file: {args.parquet}")
    if args.index < 0 or args.index >= len(df):
        raise IndexError(f"Index {args.index} out of range for {args.parquet} with {len(df)} rows")

    row = df.iloc[args.index]

    prompt = as_sequence(row["prompt"])
    if prompt and isinstance(prompt[0], dict):
        question = prompt[0].get("content", "")
    else:
        question = str(prompt)

    reward_model = as_mapping(row.get("reward_model", {}))
    ground_truth = reward_model.get("ground_truth", "")
    candidate_answers = reward_model.get("candidate_answers", [])
    extra_info = as_mapping(row.get("extra_info", {}))
    question_id = extra_info.get("question_id", args.index)

    images = as_sequence(row[args.image_key])
    if not images:
        raise ValueError(f"No images found under column '{args.image_key}'")
    image_payload = images[0]
    if not isinstance(image_payload, dict) and hasattr(image_payload, "item"):
        image_payload = image_payload.item()
    if not isinstance(image_payload, dict) or "bytes" not in image_payload:
        raise ValueError(f"Unsupported image payload: {type(image_payload)}")

    os.makedirs(args.output_dir, exist_ok=True)
    image_path = os.path.join(args.output_dir, f"{question_id}.png")
    image = Image.open(BytesIO(image_payload["bytes"]))
    image.save(image_path)

    print(f"IMAGE_PATH={image_path}")
    print(f"QUESTION={question}")
    print(f"GROUND_TRUTH={ground_truth}")
    print(f"CANDIDATE_ANSWERS={candidate_answers}")
    print(f"QUESTION_ID={question_id}")


if __name__ == "__main__":
    main()
