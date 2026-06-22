#!/usr/bin/env python3
"""Evaluate base or LoRA Qwen-VL models on locate-only Pix2Fact-DAGIG tasks."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:  # pragma: no cover - server environment has Pillow.
    Image = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter_dir", default="")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def build_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = str(example["prompt"])
    images = [p for p in example.get("images", []) if p]
    if images:
        content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
    else:
        content = [{"type": "text", "text": prompt}]
    return [{"role": "user", "content": content}]


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        min_pixels=256 * 28 * 28,
        max_pixels=args.max_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    return model, processor


def generate_one(model: Any, processor: Any, example: dict[str, Any], max_new_tokens: int) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    with torch.inference_mode():
        messages = build_messages(example)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
        return processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


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


def target_candidate_mode(ex: dict[str, Any]) -> str:
    mode = str(ex.get("coord_mode", "qwen_0_1000"))
    if mode not in {"original_pixel", "qwen_0_1000", "input_pixel"}:
        return "qwen_0_1000"
    return mode


def score_example(ex: dict[str, Any], prediction: str) -> dict[str, Any]:
    width = float(ex.get("image_width") or 1.0)
    height = float(ex.get("image_height") or 1.0)
    raw_pred = extract_locate(prediction)
    candidates = bbox_candidates(raw_pred, ex, width, height)
    gold = normalize_bbox([float(v) for v in ex.get("bbox_xyxy", [])[:4]], width, height)
    mode = target_candidate_mode(ex)
    pred = candidates.get(mode)
    score = iou(pred, gold)
    row = {
        "sample_id": ex.get("sample_id"),
        "variant": ex.get("variant"),
        "split": ex.get("split"),
        "coord_mode": mode,
        "prediction": prediction,
        "raw_locate": json.dumps(raw_pred),
        "target_bbox": json.dumps(ex.get("target_bbox")),
        "pred_bbox_original": json.dumps(pred),
        "gold_bbox_original": json.dumps(gold),
        "bbox_iou": score,
        "center_hit": center_hit(pred, gold),
        "success_iou_0_1": score >= 0.1,
        "success_iou_0_3": score >= 0.3,
        "success_iou_0_5": score >= 0.5,
        "valid_bbox": pred is not None,
        "valid_tag": bool(re.search(r"<locate>.*?</locate>", prediction, flags=re.IGNORECASE | re.DOTALL)),
        "question": ex.get("question", ""),
    }
    best_iou = 0.0
    for candidate_mode, box in candidates.items():
        candidate_iou = iou(box, gold)
        best_iou = max(best_iou, candidate_iou)
        row[f"{candidate_mode}_bbox"] = json.dumps(box)
        row[f"{candidate_mode}_iou"] = candidate_iou
        row[f"{candidate_mode}_center_hit"] = center_hit(box, gold)
    row["best_iou"] = best_iou
    row["success_iou_0_3_best"] = best_iou >= 0.3
    diagnostics = ex.get("diagnostics")
    if isinstance(diagnostics, dict):
        for key in [
            "bbox_width_at_max_pixels",
            "bbox_height_at_max_pixels",
            "bbox_area_original_ratio",
            "processor_scale_for_max_pixels",
        ]:
            row[key] = diagnostics.get(key)
    return row


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("variant", "unknown"))].append(row)
        groups["all"].append(row)
    metric_keys = [
        "valid_tag",
        "valid_bbox",
        "bbox_iou",
        "center_hit",
        "success_iou_0_1",
        "success_iou_0_3",
        "success_iou_0_5",
        "original_pixel_iou",
        "qwen_0_1000_iou",
        "input_pixel_iou",
        "best_iou",
        "success_iou_0_3_best",
    ]
    out = []
    for variant, items in groups.items():
        row = {
            "variant": variant,
            "n": len(items),
            "coord_mode": str(items[0].get("coord_mode", "")),
        }
        for key in metric_keys:
            row[key] = sum(float(item.get(key, 0.0) or 0.0) for item in items) / max(1, len(items))
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    examples = load_jsonl(args.eval_file, args.limit)
    if not examples:
        raise ValueError(f"No eval examples in {args.eval_file}")
    model, processor = load_model(args)
    rows = [score_example(ex, generate_one(model, processor, ex, args.max_new_tokens)) for ex in examples]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path = out_path.with_name(out_path.stem + "_details.csv")
    detail_jsonl = out_path.with_name(out_path.stem + "_details.jsonl")
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with detail_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = aggregate(rows)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {out_path}")
    print(f"wrote {detail_path}")
    print(f"wrote {detail_jsonl}")


if __name__ == "__main__":
    main()
