#!/usr/bin/env python3
"""Build target-context crops where the target covers about 10% of the image."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


DEFAULT_PACKAGE_NAME = "pix2fact_dagig_1k_gpt54_teacher_clean_package"
DEFAULT_MAIN_FILE = "data/pix2fact_dagig_train_AB_clean_split.jsonl"
CHAIN_SEGMENTS = ["locate", "observe", "search_decision", "search", "evidence", "answer"]


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
    parser.add_argument("--out_dir", default="data/dagig_context10")
    parser.add_argument("--main_file", default=DEFAULT_MAIN_FILE)
    parser.add_argument("--target_area_ratio", type=float, default=0.10)
    parser.add_argument("--coord_mode", choices=["qwen_0_1000", "input_pixel"], default="qwen_0_1000")
    parser.add_argument("--min_aspect", type=float, default=1 / 3)
    parser.add_argument("--max_aspect", type=float, default=3.0)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--variant_prefix", default="context10")
    parser.add_argument("--include_test", action="store_true")
    parser.add_argument("--contact_sheet_n", type=int, default=24)
    return parser.parse_args()


def load_bbox(row: dict[str, Any]) -> list[float]:
    bbox = row.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Missing bbox_xyxy for sample_id={row.get('sample_id')}")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox_xyxy for sample_id={row.get('sample_id')}: {bbox}")
    return [x1, y1, x2, y2]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def context_crop_box(
    image_width: int,
    image_height: int,
    bbox: list[float],
    target_area_ratio: float,
    min_aspect: float,
    max_aspect: float,
) -> list[int]:
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_area = bbox_width * bbox_height
    target_area_ratio = clamp(target_area_ratio, 1e-4, 1.0)
    crop_area = bbox_area / target_area_ratio

    aspect = clamp(bbox_width / bbox_height, min_aspect, max_aspect)
    crop_width = math.sqrt(crop_area * aspect)
    crop_height = crop_area / crop_width
    scale = max(1.0, bbox_width / max(crop_width, 1.0), bbox_height / max(crop_height, 1.0))
    crop_width *= scale
    crop_height *= scale
    crop_width = min(float(image_width), max(crop_width, bbox_width))
    crop_height = min(float(image_height), max(crop_height, bbox_height))

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = clamp(center_x - crop_width / 2.0, 0.0, max(0.0, image_width - crop_width))
    top = clamp(center_y - crop_height / 2.0, 0.0, max(0.0, image_height - crop_height))
    right = left + crop_width
    bottom = top + crop_height

    left_i = max(0, int(math.floor(left)))
    top_i = max(0, int(math.floor(top)))
    right_i = min(image_width, int(math.ceil(right)))
    bottom_i = min(image_height, int(math.ceil(bottom)))

    # Rounding can shave off an edge by one pixel; force containment.
    left_i = min(left_i, int(math.floor(x1)))
    top_i = min(top_i, int(math.floor(y1)))
    right_i = max(right_i, int(math.ceil(x2)))
    bottom_i = max(bottom_i, int(math.ceil(y2)))
    left_i = max(0, left_i)
    top_i = max(0, top_i)
    right_i = min(image_width, right_i)
    bottom_i = min(image_height, bottom_i)
    return [left_i, top_i, right_i, bottom_i]


def bbox_in_crop(bbox: list[float], crop_box: list[int]) -> list[float]:
    left, top, _, _ = crop_box
    x1, y1, x2, y2 = bbox
    return [x1 - left, y1 - top, x2 - left, y2 - top]


def scale_bbox_to_int(bbox: list[float], sx: float, sy: float) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy))]


def target_bbox_text(bbox_crop: list[float], crop_width: int, crop_height: int, coord_mode: str) -> str:
    if coord_mode == "qwen_0_1000":
        values = scale_bbox_to_int(bbox_crop, 1000.0 / crop_width, 1000.0 / crop_height)
    elif coord_mode == "input_pixel":
        values = [int(round(v)) for v in bbox_crop]
    else:
        raise ValueError(f"Unsupported coord_mode={coord_mode}")
    return json.dumps(values)


def locate_prompt(coord_mode: str, row: dict[str, Any]) -> str:
    if coord_mode == "qwen_0_1000":
        coord_text = "Use integer coordinates normalized to a 0-1000 image grid."
    else:
        coord_text = "Use integer pixel coordinates in the provided crop image."
    return (
        "Your task is localization only. This image is a context crop around the relevant visual clue. "
        "Do not answer the question. Do not explain. Locate the exact visual target referred to by the question. "
        "Return exactly one XML-like section: <locate>[x1, y1, x2, y2]</locate>. "
        f"{coord_text}\n\n"
        f"Question: {str(row.get('question', '')).strip()}"
    )


def chain_prompt(coord_mode: str, row: dict[str, Any]) -> str:
    if coord_mode == "qwen_0_1000":
        coord_text = "The <locate> coordinates must use integer coordinates normalized to a 0-1000 image grid."
    else:
        coord_text = "The <locate> coordinates must use integer pixel coordinates in the provided crop image."
    return (
        "You are a multimodal search agent. This image is a context crop around the relevant visual clue. "
        "Return exactly these XML-like sections: <locate>, <observe>, <search_decision>, <search>, <evidence>, <answer>. "
        f"{coord_text}\n\n"
        f"Question: {str(row.get('question', '')).strip()}"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path} n={len(rows)}")


def format_chain_target(segments: dict[str, str]) -> str:
    return "\n".join(f"<{name}>\n{segments[name]}\n</{name}>" for name in CHAIN_SEGMENTS)


def crop_diagnostics(row: dict[str, Any], bbox: list[float], crop_box: list[int], bbox_crop: list[float]) -> dict[str, Any]:
    crop_width = crop_box[2] - crop_box[0]
    crop_height = crop_box[3] - crop_box[1]
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    crop_area = max(1, crop_width * crop_height)
    bbox_area = bbox_width * bbox_height
    return {
        "crop_box_xyxy_original": crop_box,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "bbox_in_crop_xyxy": [round(v, 3) for v in bbox_crop],
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "target_area_ratio_actual": bbox_area / crop_area,
        "target_width_ratio_actual": bbox_width / max(1, crop_width),
        "target_height_ratio_actual": bbox_height / max(1, crop_height),
        "original_image_width": row.get("image_width"),
        "original_image_height": row.get("image_height"),
    }


def make_examples(
    chain_builder: Any,
    package_dir: Path,
    row: dict[str, Any],
    image_out_dir: Path,
    target_area_ratio: float,
    coord_mode: str,
    min_aspect: float,
    max_aspect: float,
    jpeg_quality: int,
    variant_prefix: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    full_image_path = chain_builder.resolve_image_path(package_dir, row, "full_original")
    bbox = load_bbox(row)
    with Image.open(full_image_path) as image:
        image = image.convert("RGB")
        crop_box = context_crop_box(image.width, image.height, bbox, target_area_ratio, min_aspect, max_aspect)
        crop = image.crop(tuple(crop_box))
        crop_width, crop_height = crop.size
        crop_path = image_out_dir / f"{row.get('sample_id')}_context10.jpg"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path, quality=jpeg_quality)

    bbox_crop = bbox_in_crop(bbox, crop_box)
    locate_text = target_bbox_text(bbox_crop, crop_width, crop_height, coord_mode)
    diagnostics = crop_diagnostics(row, bbox, crop_box, bbox_crop)
    base = chain_builder.make_example(package_dir, row, "dagig_sft", min_process_weight=0.05)

    locate_target = f"<locate>\n{locate_text}\n</locate>"
    locate_prompt_text = locate_prompt(coord_mode, row)
    locate_variant = f"{variant_prefix}_locate_{coord_mode}"
    common = {
        "sample_id": row.get("sample_id"),
        "split": row.get("split", "train"),
        "coord_mode": coord_mode,
        "images": [str(crop_path.resolve())],
        "context_crop_path": str(crop_path.resolve()),
        "source_full_original_path": str(Path(full_image_path).resolve()),
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "bbox_xyxy": [int(round(v)) for v in bbox],
        "bbox_in_context_crop_xyxy": [round(v, 3) for v in bbox_crop],
        "target_bbox": json.loads(locate_text),
        "diagnostics": diagnostics,
    }
    locate_example = {
        **common,
        "variant": locate_variant,
        "task_type": "context_crop_locate_only",
        "prompt": locate_prompt_text,
        "target": locate_target,
        "target_segments": {"locate": locate_text},
        "segment_weights": {"locate": 1.0},
        "loss_weight": 1.0,
        "messages": [
            {"role": "user", "content": locate_prompt_text},
            {"role": "assistant", "content": locate_target},
        ],
    }

    chain_segments = {"locate": locate_text, **dict(base["target_segments"])}
    chain_target = format_chain_target(chain_segments)
    chain_prompt_text = chain_prompt(coord_mode, row)
    chain_weights = {"locate": float(base["segment_weights"].get("observe", 1.0)), **dict(base["segment_weights"])}
    chain_example = {
        **base,
        **common,
        "variant": f"{variant_prefix}_dagig_sft_{coord_mode}",
        "task_type": "context_crop_dagig_chain",
        "input_mode": "target_context_crop",
        "prompt": chain_prompt_text,
        "target": chain_target,
        "target_segments": chain_segments,
        "segment_weights": chain_weights,
        "loss_weight": sum(float(v) for v in chain_weights.values()) / len(chain_weights),
        "messages": [
            {"role": "user", "content": chain_prompt_text},
            {"role": "assistant", "content": chain_target},
        ],
        "crop_image_path": str(crop_path.resolve()),
        "full_image_path": str(crop_path.resolve()),
    }
    return locate_example, chain_example, diagnostics


def summarize(diagnostics: list[dict[str, Any]], out_path: Path) -> None:
    def mean(key: str) -> float:
        vals = [float(item[key]) for item in diagnostics]
        return sum(vals) / max(1, len(vals))

    def rate_close(low: float, high: float) -> float:
        vals = [float(item["target_area_ratio_actual"]) for item in diagnostics]
        return sum(low <= value <= high for value in vals) / max(1, len(vals))

    def rate_min_side_at_processor_below(threshold: float) -> float:
        # With a 512x512-ish processor budget, a 10% area target has a large side.
        vals = []
        for item in diagnostics:
            crop_w = float(item["crop_width"])
            crop_h = float(item["crop_height"])
            scale = min(1.0, math.sqrt((512 * 512) / max(1.0, crop_w * crop_h)))
            vals.append(min(float(item["bbox_width"]) * scale, float(item["bbox_height"]) * scale))
        return sum(value < threshold for value in vals) / max(1, len(vals))

    ratios = sorted(float(item["target_area_ratio_actual"]) for item in diagnostics)
    summary = {
        "n": len(diagnostics),
        "target_area_ratio_mean": mean("target_area_ratio_actual"),
        "target_area_ratio_min": ratios[0] if ratios else 0.0,
        "target_area_ratio_p10": ratios[int(0.10 * (len(ratios) - 1))] if ratios else 0.0,
        "target_area_ratio_p50": ratios[int(0.50 * (len(ratios) - 1))] if ratios else 0.0,
        "target_area_ratio_p90": ratios[int(0.90 * (len(ratios) - 1))] if ratios else 0.0,
        "target_area_ratio_max": ratios[-1] if ratios else 0.0,
        "rate_area_ratio_0_08_to_0_12": rate_close(0.08, 0.12),
        "crop_width_mean": mean("crop_width"),
        "crop_height_mean": mean("crop_height"),
        "target_width_ratio_mean": mean("target_width_ratio_actual"),
        "target_height_ratio_mean": mean("target_height_ratio_actual"),
        "rate_min_side_at_262k_lt_24": rate_min_side_at_processor_below(24),
        "rate_min_side_at_262k_lt_32": rate_min_side_at_processor_below(32),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


def make_contact_sheet(rows: list[dict[str, Any]], out_path: Path, n: int) -> None:
    if n <= 0 or not rows:
        return
    selected = rows[: min(n, len(rows))]
    thumb_w, thumb_h = 240, 240
    cols = min(6, len(selected))
    rows_n = math.ceil(len(selected) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows_n * thumb_h), "white")
    for idx, row in enumerate(selected):
        image = Image.open(row["context_crop_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        x1, y1, x2, y2 = row["bbox_in_context_crop_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=max(2, image.width // 200))
        image.thumbnail((thumb_w, thumb_h))
        canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
        canvas.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
        sheet.paste(canvas, ((idx % cols) * thumb_w, (idx // cols) * thumb_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=95)
    print(f"wrote {out_path}")


def main() -> None:
    args = parse_args()
    if not 0 < args.target_area_ratio <= 1:
        raise ValueError("--target_area_ratio must be in (0, 1]")
    chain_builder = load_chain_builder()
    package_dir = Path(args.package_dir).expanduser().resolve()
    main_path = Path(args.main_file)
    if not main_path.is_absolute():
        main_path = package_dir / main_path
    source_rows = chain_builder.load_jsonl(main_path)
    out_dir = Path(args.out_dir).expanduser().resolve()
    image_out_dir = out_dir / "images" / f"{args.variant_prefix}_crop_original"

    locate_splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    chain_splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    diagnostics = []
    all_locate_rows: list[dict[str, Any]] = []
    for row in source_rows:
        split = str(row.get("split", "train"))
        if split not in locate_splits:
            continue
        locate_ex, chain_ex, diag = make_examples(
            chain_builder,
            package_dir,
            row,
            image_out_dir,
            args.target_area_ratio,
            args.coord_mode,
            args.min_aspect,
            args.max_aspect,
            args.jpeg_quality,
            args.variant_prefix,
        )
        locate_splits[split].append(locate_ex)
        chain_splits[split].append(chain_ex)
        diagnostics.append({"sample_id": row.get("sample_id"), "split": split, **diag})
        all_locate_rows.append(locate_ex)

    locate_variant = f"{args.variant_prefix}_locate_{args.coord_mode}"
    chain_variant = f"{args.variant_prefix}_dagig_sft_{args.coord_mode}"
    write_jsonl(out_dir / f"{locate_variant}_train.jsonl", locate_splits["train"])
    write_jsonl(out_dir / f"{locate_variant}_dev.jsonl", locate_splits["dev"])
    write_jsonl(out_dir / f"{chain_variant}_train.jsonl", chain_splits["train"])
    write_jsonl(out_dir / f"{chain_variant}_dev.jsonl", chain_splits["dev"])
    if args.include_test:
        write_jsonl(out_dir / f"{locate_variant}_test.jsonl", locate_splits["test"])
        write_jsonl(out_dir / f"{chain_variant}_test.jsonl", chain_splits["test"])
    diag_path = out_dir / f"{args.variant_prefix}_crop_diagnostics.jsonl"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with diag_path.open("w", encoding="utf-8") as f:
        for item in diagnostics:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {diag_path} n={len(diagnostics)}")
    summarize(diagnostics, out_dir / f"{args.variant_prefix}_crop_summary.json")
    make_contact_sheet(all_locate_rows, out_dir / f"{args.variant_prefix}_contact_sheet.jpg", args.contact_sheet_n)


if __name__ == "__main__":
    main()
