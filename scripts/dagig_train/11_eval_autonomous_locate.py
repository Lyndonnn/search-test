#!/usr/bin/env python3
"""Evaluate <locate> predictions from autonomous full-image chain outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional diagnostic path.
    Image = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples_jsonl", required=True)
    parser.add_argument("--details_jsonl", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--details_csv", default="")
    parser.add_argument(
        "--coord_mode",
        choices=["original_pixel", "qwen_0_1000", "input_pixel", "best"],
        default="original_pixel",
        help="Coordinate interpretation used for the primary bbox_iou columns.",
    )
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_locate(text: str) -> list[float] | None:
    match = re.search(r"<locate>\s*(.*?)\s*</locate>", text, flags=re.IGNORECASE | re.DOTALL)
    raw = match.group(1) if match else text
    nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
    if len(nums) < 4:
        return None
    return [float(v) for v in nums[:4]]


def normalize_bbox(bbox: list[float] | None, width: float, height: float) -> list[float] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted([max(0.0, min(width, x1)), max(0.0, min(width, x2))])
    y1, y2 = sorted([max(0.0, min(height, y1)), max(0.0, min(height, y2))])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def scale_bbox(bbox: list[float] | None, sx: float, sy: float) -> list[float] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def input_image_size(ex: dict[str, Any]) -> tuple[float, float] | None:
    if Image is None:
        return None
    image_path = ex.get("full_image_path")
    if not image_path:
        images = ex.get("images") or []
        image_path = images[0] if images else ""
    if not image_path:
        return None
    try:
        with Image.open(image_path) as image:
            width, height = image.size
        return float(width), float(height)
    except Exception:
        return None


def bbox_candidates(raw_bbox: list[float] | None, ex: dict[str, Any], width: float, height: float) -> dict[str, list[float] | None]:
    candidates: dict[str, list[float] | None] = {
        "original_pixel": normalize_bbox(raw_bbox, width, height),
        "qwen_0_1000": normalize_bbox(scale_bbox(raw_bbox, width / 1000.0, height / 1000.0), width, height),
        "input_pixel": None,
    }
    size = input_image_size(ex)
    if size:
        input_width, input_height = size
        candidates["input_pixel"] = normalize_bbox(
            scale_bbox(raw_bbox, width / input_width, height / input_height),
            width,
            height,
        )
    return candidates


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def center_hit(pred: list[float] | None, gold: list[float] | None) -> bool:
    if pred is None or gold is None:
        return False
    cx = (pred[0] + pred[2]) / 2.0
    cy = (pred[1] + pred[3]) / 2.0
    return gold[0] <= cx <= gold[2] and gold[1] <= cy <= gold[3]


def main() -> None:
    args = parse_args()
    examples = {str(row.get("sample_id")): row for row in load_jsonl(args.examples_jsonl)}
    details = load_jsonl(args.details_jsonl)
    rows = []
    for detail in details:
        sample_id = str(detail.get("sample_id"))
        ex = examples.get(sample_id)
        if not ex:
            continue
        width = float(ex.get("image_width") or 1.0)
        height = float(ex.get("image_height") or 1.0)
        raw_pred = extract_locate(str(detail.get("prediction", "")))
        candidates = bbox_candidates(raw_pred, ex, width, height)
        gold = normalize_bbox([float(v) for v in ex.get("bbox_xyxy", [])[:4]], width, height)
        candidate_scores = {mode: iou(box, gold) for mode, box in candidates.items()}
        candidate_centers = {mode: center_hit(box, gold) for mode, box in candidates.items()}
        coord_mode = args.coord_mode
        if coord_mode == "best":
            coord_mode = max(candidate_scores, key=candidate_scores.get)
        pred = candidates.get(coord_mode)
        score = iou(pred, gold)
        row = {
            "sample_id": sample_id,
            "variant": detail.get("variant"),
            "coord_mode": coord_mode,
            "raw_locate": json.dumps(raw_pred),
            "pred_bbox": json.dumps(pred),
            "gold_bbox": json.dumps(gold),
            "bbox_iou": score,
            "center_hit": center_hit(pred, gold),
            "locate_success_iou_0_3": score >= 0.3,
            "valid_bbox": pred is not None,
        }
        for mode in ["original_pixel", "qwen_0_1000", "input_pixel"]:
            row[f"{mode}_bbox"] = json.dumps(candidates.get(mode))
            row[f"{mode}_iou"] = candidate_scores.get(mode, 0.0)
            row[f"{mode}_center_hit"] = candidate_centers.get(mode, False)
        row["best_iou"] = max(candidate_scores.values()) if candidate_scores else 0.0
        rows.append(row)
    if not rows:
        raise ValueError("No rows evaluated")
    summary = {
        "variant": str(rows[0].get("variant")),
        "n": len(rows),
        "coord_mode": args.coord_mode,
        "valid_bbox_rate": sum(float(r["valid_bbox"]) for r in rows) / len(rows),
        "mean_iou": sum(float(r["bbox_iou"]) for r in rows) / len(rows),
        "center_hit_rate": sum(float(r["center_hit"]) for r in rows) / len(rows),
        "locate_success_iou_0_3": sum(float(r["locate_success_iou_0_3"]) for r in rows) / len(rows),
        "mean_iou_original_pixel": sum(float(r["original_pixel_iou"]) for r in rows) / len(rows),
        "mean_iou_qwen_0_1000": sum(float(r["qwen_0_1000_iou"]) for r in rows) / len(rows),
        "mean_iou_input_pixel": sum(float(r["input_pixel_iou"]) for r in rows) / len(rows),
        "mean_best_iou": sum(float(r["best_iou"]) for r in rows) / len(rows),
        "success_iou_0_3_original_pixel": sum(float(r["original_pixel_iou"]) >= 0.3 for r in rows) / len(rows),
        "success_iou_0_3_qwen_0_1000": sum(float(r["qwen_0_1000_iou"]) >= 0.3 for r in rows) / len(rows),
        "success_iou_0_3_input_pixel": sum(float(r["input_pixel_iou"]) >= 0.3 for r in rows) / len(rows),
        "success_iou_0_3_best": sum(float(r["best_iou"]) >= 0.3 for r in rows) / len(rows),
    }
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    detail_path = Path(args.details_csv) if args.details_csv else out_path.with_name(out_path.stem + "_details.csv")
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")
    print(f"wrote {detail_path}")


if __name__ == "__main__":
    main()
