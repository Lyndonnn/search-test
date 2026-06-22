#!/usr/bin/env python3
"""Shared helpers for the RN03_10 grounded-action pipeline."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image, ImageDraw


PACKAGE_DIR = Path("data/dagig_rn03_10_ground_expr_v3_full")
RESULT_ROOT = Path("results/dagig_rn03_10_grounded")
GROUND_EXPR_PREFIX = "pix2fact_dagig_rn03_10_ground_expr_gpt54_v3"
HARD_FILES = {
    "train": f"data/{GROUND_EXPR_PREFIX}_train.jsonl",
    "dev": f"data/{GROUND_EXPR_PREFIX}_dev.jsonl",
    "test": f"data/{GROUND_EXPR_PREFIX}_test.jsonl",
    "all": f"data/{GROUND_EXPR_PREFIX}_train_AB_clean_split.jsonl",
}
REVIEW_FILES = {
    "all": f"data/{GROUND_EXPR_PREFIX}_review_needed.jsonl",
    "train": f"data/{GROUND_EXPR_PREFIX}_review_needed_train.jsonl",
    "dev": f"data/{GROUND_EXPR_PREFIX}_review_needed_dev.jsonl",
    "test": f"data/{GROUND_EXPR_PREFIX}_review_needed_test.jsonl",
}
FORBIDDEN_IMAGE_TERMS = (
    "red",
    "redbox",
    "red_box",
    "annotation",
    "annotated",
    "vis",
    "contact_sheet",
    "bbox",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DINO_CONFIG = "third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_WEIGHTS = "third_party/GroundingDINO_weights/groundingdino_swint_ogc.pth"


def load_jsonl(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected object at {path}:{line_no}")
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k)) for k in fieldnames})


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def md_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    if not rows:
        return ""
    columns = columns or list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        vals = [str(csv_value(row.get(col, ""))).replace("\n", " ") for col in columns]
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def package_path(package_root: str | Path, relpath: str | Path) -> Path:
    rel = Path(str(relpath))
    return rel if rel.is_absolute() else Path(package_root) / rel


def row_image_path(package_root: str | Path, row: dict[str, Any]) -> Path:
    rel = row.get("clean_rn_image_relpath_from_package")
    if not rel:
        gt = row.get("ground_tool_gold")
        if isinstance(gt, dict):
            rel = gt.get("rn_image_relpath_from_package")
    if not rel:
        raise KeyError(f"Missing clean RN image relpath for sample_id={row.get('sample_id')}")
    return package_path(package_root, str(rel))


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def open_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def has_forbidden_image_term(path_text: str) -> bool:
    lower = str(path_text).lower()
    return any(term in lower for term in FORBIDDEN_IMAGE_TERMS)


def hard_pass(row: dict[str, Any]) -> bool:
    quality = row.get("ground_expression_quality")
    return isinstance(quality, dict) and quality.get("hard_pass") is True


def get_gold_bbox(row: dict[str, Any]) -> list[float]:
    gt = row.get("ground_tool_gold")
    bbox = gt.get("bbox_pixel_xyxy") if isinstance(gt, dict) else None
    if not valid_bbox_like(bbox):
        bbox = row.get("bbox_in_rn_pixel_xyxy") or row.get("bbox_xyxy")
    if not valid_bbox_like(bbox):
        raise ValueError(f"Missing gold bbox for sample_id={row.get('sample_id')}")
    return [float(v) for v in bbox[:4]]


def valid_bbox_like(value: Any) -> bool:
    if not isinstance(value, list) or len(value) < 4:
        return False
    try:
        [float(v) for v in value[:4]]
        return True
    except (TypeError, ValueError):
        return False


def normalize_bbox(bbox: list[float] | None, width: float, height: float) -> list[float] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    x1, x2 = sorted([max(0.0, min(width, x1)), max(0.0, min(width, x2))])
    y1, y2 = sorted([max(0.0, min(height, y1)), max(0.0, min(height, y2))])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_area(bbox: list[float] | None) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_iou(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    denom = bbox_area(a) + bbox_area(b) - inter
    return inter / denom if denom > 0 else 0.0


def center_hit(pred: list[float] | None, gold: list[float] | None) -> bool:
    if pred is None or gold is None:
        return False
    cx = (pred[0] + pred[2]) / 2.0
    cy = (pred[1] + pred[3]) / 2.0
    return gold[0] <= cx <= gold[2] and gold[1] <= cy <= gold[3]


def area_ratio(pred: list[float] | None, gold: list[float] | None) -> float | None:
    gold_area = bbox_area(gold)
    if pred is None or gold_area <= 0:
        return None
    return bbox_area(pred) / gold_area


def is_extreme_box(pred: list[float] | None, gold: list[float] | None, width: int, height: int) -> bool:
    if pred is None:
        return False
    ratio = area_ratio(pred, gold)
    image_ratio = bbox_area(pred) / max(1.0, float(width * height))
    return bool(
        ratio is None
        or ratio < 0.02
        or ratio > 25.0
        or image_ratio < 0.0001
        or image_ratio > 0.75
    )


def safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text[:120] or "sample"


def draw_boxes(
    image_path: str | Path,
    gold: list[float] | None,
    pred: list[float] | None,
    label: str = "",
) -> Image.Image:
    image = open_rgb(image_path)
    draw = ImageDraw.Draw(image)
    if gold is not None:
        draw.rectangle(gold, outline=(255, 0, 0), width=4)
    if pred is not None:
        draw.rectangle(pred, outline=(0, 90, 255), width=4)
    if label:
        draw.rectangle([0, 0, image.width, min(image.height, 18)], fill=(255, 255, 255))
        draw.text((4, 3), label[:180], fill=(0, 0, 0))
    return image


def save_crop(image_path: str | Path, bbox: list[float] | None, out_path: str | Path) -> str:
    if bbox is None:
        return ""
    image = open_rgb(image_path)
    box = [
        int(round(max(0.0, min(float(image.width), bbox[0])))),
        int(round(max(0.0, min(float(image.height), bbox[1])))),
        int(round(max(0.0, min(float(image.width), bbox[2])))),
        int(round(max(0.0, min(float(image.height), bbox[3])))),
    ]
    if box[2] <= box[0] or box[3] <= box[1]:
        return ""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.crop(tuple(box)).save(out)
    return str(out)


def load_groundingdino_model(config: str, weights: str, device: str):
    from groundingdino.util.inference import load_model

    return load_model(config, weights, device=device)


def predict_groundingdino(
    model: Any,
    image_path: str | Path,
    text: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> dict[str, Any]:
    import torch
    from torchvision.ops import box_convert
    from groundingdino.util.inference import load_image, predict

    image_source, image = load_image(str(image_path))
    height, width = image_source.shape[:2]
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )
    predictions: list[dict[str, Any]] = []
    if len(boxes):
        scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
        xyxy = box_convert(boxes=boxes * scale, in_fmt="cxcywh", out_fmt="xyxy")
        for idx, (box_norm, box_xyxy, score, phrase) in enumerate(zip(boxes, xyxy, logits, phrases)):
            raw_box = [float(v) for v in box_xyxy.detach().cpu().tolist()]
            pred = normalize_bbox(raw_box, width, height)
            predictions.append(
                {
                    "index": idx,
                    "phrase": str(phrase),
                    "score": float(score),
                    "box_cxcywh_norm": [float(v) for v in box_norm.detach().cpu().tolist()],
                    "box_xyxy": pred,
                    "box_xyxy_raw": raw_box,
                }
            )
        predictions.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return {"width": width, "height": height, "predictions": predictions}


def evaluate_grounding_rows(
    rows: list[dict[str, Any]],
    package_root: str | Path,
    model: Any,
    box_threshold: float,
    text_threshold: float,
    device: str,
    out_dir: str | Path | None = None,
    save_images: bool = True,
) -> list[dict[str, Any]]:
    out = Path(out_dir) if out_dir is not None else None
    crops_dir = out / "crops" if out else None
    vis_dir = out / "vis" if out else None
    if save_images and out:
        crops_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        sample_id = str(row.get("sample_id", idx))
        image_path = row_image_path(package_root, row)
        width, height = image_size(image_path)
        gold = normalize_bbox(get_gold_bbox(row), width, height)
        expression = str(row.get("ground_expression", "")).strip()
        dino = predict_groundingdino(model, image_path, expression, box_threshold, text_threshold, device)
        top = dino["predictions"][0] if dino["predictions"] else None
        pred = top.get("box_xyxy") if isinstance(top, dict) else None
        score = float(top.get("score")) if isinstance(top, dict) else None
        iou = bbox_iou(pred, gold)
        hit = center_hit(pred, gold)
        ratio = area_ratio(pred, gold)
        extreme = is_extreme_box(pred, gold, width, height)
        stem = safe_name(sample_id)
        crop_path = ""
        vis_path = ""
        if save_images and out:
            if pred is not None and crops_dir is not None:
                crop_path = save_crop(image_path, pred, crops_dir / f"{stem}.jpg")
            if vis_dir is not None:
                label = f"{sample_id} score={score if score is not None else 'NA'} iou={iou:.3f}"
                vis = draw_boxes(image_path, gold=gold, pred=pred, label=label)
                vis_path = str(vis_dir / f"{stem}.jpg")
                vis.save(vis_path)
        result = {
            "sample_id": sample_id,
            "split": row.get("split"),
            "question_type": row.get("question_type"),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "ground_expression": expression,
            "image_relpath": row.get("clean_rn_image_relpath_from_package"),
            "image_path": str(image_path),
            "image_width": width,
            "image_height": height,
            "gold_bbox_xyxy": gold,
            "pred_bbox_xyxy": pred,
            "best_score": score,
            "best_phrase": top.get("phrase") if isinstance(top, dict) else "",
            "num_detections": len(dino["predictions"]),
            "detected": pred is not None,
            "iou": iou,
            "iou_ge_0_1": iou >= 0.1,
            "iou_ge_0_3": iou >= 0.3,
            "iou_ge_0_5": iou >= 0.5,
            "center_hit": hit,
            "pred_gold_area_ratio": ratio,
            "extreme_box": extreme,
            "all_predictions": dino["predictions"],
            "crop_path": crop_path,
            "vis_path": vis_path,
        }
        results.append(result)
    return results


def aggregate_grounding_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    detected = [r for r in rows if r.get("detected")]
    ious = [float(r.get("iou", 0.0) or 0.0) for r in rows]
    scores = [float(r["best_score"]) for r in detected if r.get("best_score") is not None]
    ratios = [float(r["pred_gold_area_ratio"]) for r in detected if r.get("pred_gold_area_ratio") is not None]

    def rate(key: str) -> float:
        return sum(float(bool(r.get(key))) for r in rows) / n

    return {
        "n": n,
        "detected": len(detected),
        "detection_rate": len(detected) / n,
        "no_detection_rate": 1.0 - len(detected) / n,
        "mean_iou": sum(ious) / n,
        "median_iou": median(ious),
        "iou_ge_0_1": rate("iou_ge_0_1"),
        "iou_ge_0_3": rate("iou_ge_0_3"),
        "iou_ge_0_5": rate("iou_ge_0_5"),
        "center_hit_rate": rate("center_hit"),
        "mean_best_score": sum(scores) / len(scores) if scores else 0.0,
        "median_best_score": median(scores) if scores else 0.0,
        "mean_pred_gold_area_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
        "median_pred_gold_area_ratio": median(ratios) if ratios else 0.0,
        "extreme_box_rate": sum(float(bool(r.get("extreme_box"))) for r in rows) / n,
    }


def write_grounding_outputs(out_dir: str | Path, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "grounding_results.jsonl", rows)
    write_csv(out / "grounding_results.csv", rows)
    write_json(out / "metrics.json", metrics)
    metric_rows = [{"metric": k, "value": v} for k, v in metrics.items()]
    md = "# GroundingDINO Metrics\n\n" + md_table(metric_rows, ["metric", "value"]) + "\n"
    (out / "metrics.md").write_text(md, encoding="utf-8")


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip().split()


def token_f1(pred: str, target: str) -> float:
    pred_tokens = tokenize(pred)
    target_tokens = tokenize(target)
    if not pred_tokens or not target_tokens:
        return 0.0
    common = 0
    target_counts: dict[str, int] = {}
    for tok in target_tokens:
        target_counts[tok] = target_counts.get(tok, 0) + 1
    for tok in pred_tokens:
        if target_counts.get(tok, 0) > 0:
            common += 1
            target_counts[tok] -= 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", str(text), flags=re.I | re.S)
    return match.group(1).strip() if match else ""


def extract_segments(text: str, tags: list[str]) -> dict[str, str]:
    return {tag: extract_tag(text, tag) for tag in tags}


def evidence_text(row: dict[str, Any]) -> str:
    teacher = row.get("gpt54_teacher")
    if isinstance(teacher, dict):
        quote = str(teacher.get("supporting_evidence_quote") or "").strip()
        if quote:
            return quote
    for key in ("evidence", "evidence_text", "supporting_evidence"):
        text = str(row.get(key) or "").strip()
        if text:
            return text
    evidences = row.get("evidences")
    if isinstance(evidences, list):
        for item in evidences:
            if isinstance(item, dict) and item.get("answer_supported"):
                text = str(item.get("text") or "").strip()
                if text:
                    return text
        for item in evidences:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                if text:
                    return text
    return ""


def select_rows(rows: list[dict[str, Any]], count: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if len(rows) <= count:
        return list(rows)
    return rng.sample(rows, count)

