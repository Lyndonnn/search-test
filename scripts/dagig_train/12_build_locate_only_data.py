#!/usr/bin/env python3
"""Build locate-only full-image SFT/eval data for Pix2Fact-DAGIG."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:  # pragma: no cover - server environment has Pillow.
    Image = None


DEFAULT_PACKAGE_NAME = "pix2fact_dagig_1k_gpt54_teacher_clean_package"
DEFAULT_MAIN_FILE = "data/pix2fact_dagig_train_AB_clean_split.jsonl"


def load_chain_builder() -> Any:
    path = Path(__file__).with_name("01_build_sft_data.py")
    spec = importlib.util.spec_from_file_location("dagig_chain_builder", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package_dir", default=f"data/{DEFAULT_PACKAGE_NAME}")
    parser.add_argument("--out_dir", default="data/dagig_locate_only")
    parser.add_argument("--main_file", default=DEFAULT_MAIN_FILE)
    parser.add_argument(
        "--coord_mode",
        choices=["original_pixel", "qwen_0_1000", "input_pixel"],
        default="qwen_0_1000",
        help="Coordinate convention used in the supervised <locate> target.",
    )
    parser.add_argument("--variant_prefix", default="locate_only")
    parser.add_argument("--include_test", action="store_true")
    parser.add_argument("--max_pixels", type=int, default=512 * 512, help="Diagnostic processor budget, not used to rewrite images.")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path} n={len(rows)}")


def bbox_xyxy(row: dict[str, Any]) -> list[float]:
    bbox = row.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Missing bbox_xyxy for sample_id={row.get('sample_id')}")
    return [float(v) for v in bbox]


def image_size(path: str) -> tuple[float, float]:
    if Image is None:
        raise ImportError("Pillow is required to inspect image dimensions")
    with Image.open(path) as image:
        width, height = image.size
    return float(width), float(height)


def scale_bbox(bbox: list[float], sx: float, sy: float) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy))]


def target_bbox(row: dict[str, Any], full_image: str, coord_mode: str) -> list[int]:
    bbox = bbox_xyxy(row)
    width = float(row.get("image_width") or 1.0)
    height = float(row.get("image_height") or 1.0)
    if coord_mode == "original_pixel":
        return [int(round(v)) for v in bbox]
    if coord_mode == "qwen_0_1000":
        return scale_bbox(bbox, 1000.0 / width, 1000.0 / height)
    if coord_mode == "input_pixel":
        input_width, input_height = image_size(full_image)
        return scale_bbox(bbox, input_width / width, input_height / height)
    raise ValueError(f"Unsupported coord_mode={coord_mode}")


def target_prompt(coord_mode: str, row: dict[str, Any]) -> str:
    if coord_mode == "qwen_0_1000":
        coord_text = "Use integer coordinates normalized to a 0-1000 image grid."
    elif coord_mode == "input_pixel":
        coord_text = "Use integer pixel coordinates in the provided resized image."
    else:
        coord_text = "Use integer pixel coordinates in the original full image."
    return (
        "Your task is localization only. Do not answer the question. Do not explain. "
        "Identify the image region containing the visual clue referred to by the question. "
        "Always return exactly one XML-like section and four integers: "
        "<locate>[x1, y1, x2, y2]</locate>. "
        f"{coord_text} If uncertain, still give your best bounding box.\n\n"
        f"Question: {str(row.get('question', '')).strip()}"
    )


def processed_target_stats(row: dict[str, Any], full_image: str, max_pixels: int) -> dict[str, float]:
    width = float(row.get("image_width") or 1.0)
    height = float(row.get("image_height") or 1.0)
    input_width, input_height = image_size(full_image)
    x1, y1, x2, y2 = bbox_xyxy(row)
    box_width_input = max(0.0, (x2 - x1) * input_width / width)
    box_height_input = max(0.0, (y2 - y1) * input_height / height)
    processor_scale = min(1.0, math.sqrt(float(max_pixels) / max(1.0, input_width * input_height)))
    return {
        "full_model_input_width": input_width,
        "full_model_input_height": input_height,
        "bbox_width_original": max(0.0, x2 - x1),
        "bbox_height_original": max(0.0, y2 - y1),
        "bbox_area_original_ratio": max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, width * height),
        "bbox_width_full_model_input": box_width_input,
        "bbox_height_full_model_input": box_height_input,
        "bbox_width_at_max_pixels": box_width_input * processor_scale,
        "bbox_height_at_max_pixels": box_height_input * processor_scale,
        "processor_scale_for_max_pixels": processor_scale,
    }


def make_example(chain_builder: Any, package_dir: Path, row: dict[str, Any], coord_mode: str, variant: str, max_pixels: int) -> dict[str, Any]:
    full_image = chain_builder.resolve_image_path(package_dir, row, "full_model_input")
    bbox = target_bbox(row, full_image, coord_mode)
    target = f"<locate>\n{json.dumps(bbox)}\n</locate>"
    prompt = target_prompt(coord_mode, row)
    return {
        "sample_id": row.get("sample_id"),
        "variant": variant,
        "split": row.get("split", "train"),
        "task_type": "locate_only",
        "coord_mode": coord_mode,
        "prompt": prompt,
        "target": target,
        "target_segments": {"locate": json.dumps(bbox)},
        "segment_weights": {"locate": 1.0},
        "loss_weight": 1.0,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "images": [full_image],
        "full_image_path": full_image,
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "bbox_xyxy": [int(round(v)) for v in bbox_xyxy(row)],
        "target_bbox": bbox,
        "image_width": row.get("image_width"),
        "image_height": row.get("image_height"),
        "image_paths": row.get("image_paths", {}),
        "diagnostics": processed_target_stats(row, full_image, max_pixels),
    }


def summarize(rows: list[dict[str, Any]], out_path: Path) -> None:
    def mean(key: str) -> float:
        vals = [float(row["diagnostics"][key]) for row in rows]
        return sum(vals) / max(1, len(vals))

    def rate_min_side_below(threshold: float) -> float:
        vals = [
            min(float(row["diagnostics"]["bbox_width_at_max_pixels"]), float(row["diagnostics"]["bbox_height_at_max_pixels"]))
            for row in rows
        ]
        return sum(v < threshold for v in vals) / max(1, len(vals))

    summary = {
        "n": len(rows),
        "mean_bbox_area_original_ratio": mean("bbox_area_original_ratio"),
        "mean_bbox_width_at_max_pixels": mean("bbox_width_at_max_pixels"),
        "mean_bbox_height_at_max_pixels": mean("bbox_height_at_max_pixels"),
        "rate_min_side_at_max_pixels_lt_8": rate_min_side_below(8),
        "rate_min_side_at_max_pixels_lt_16": rate_min_side_below(16),
        "rate_min_side_at_max_pixels_lt_24": rate_min_side_below(24),
        "rate_min_side_at_max_pixels_lt_32": rate_min_side_below(32),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    args = parse_args()
    chain_builder = load_chain_builder()
    package_dir = Path(args.package_dir).expanduser().resolve()
    main_path = Path(args.main_file)
    if not main_path.is_absolute():
        main_path = package_dir / main_path
    source_rows = chain_builder.load_jsonl(main_path)
    variant = f"{args.variant_prefix}_{args.coord_mode}"
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for row in source_rows:
        split = str(row.get("split", "train"))
        if split not in splits:
            continue
        splits[split].append(make_example(chain_builder, package_dir, row, args.coord_mode, variant, args.max_pixels))
    out_dir = Path(args.out_dir).expanduser().resolve()
    write_jsonl(out_dir / f"{variant}_train.jsonl", splits["train"])
    write_jsonl(out_dir / f"{variant}_dev.jsonl", splits["dev"])
    if args.include_test:
        write_jsonl(out_dir / f"{variant}_test.jsonl", splits["test"])
    summarize(splits["train"] + splits["dev"] + splits["test"], out_dir / f"{variant}_diagnostics.json")


if __name__ == "__main__":
    main()
