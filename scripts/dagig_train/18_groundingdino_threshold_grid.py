#!/usr/bin/env python3
"""Grid-search GroundingDINO thresholds on the dev hard-pass set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grounded_pipeline_utils import (
    DINO_CONFIG,
    DINO_WEIGHTS,
    aggregate_grounding_metrics,
    area_ratio,
    bbox_iou,
    center_hit,
    evaluate_grounding_rows,
    is_extreme_box,
    load_groundingdino_model,
    load_jsonl,
    write_csv,
    write_json,
)


BOX_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
TEXT_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--package_root", required=True)
    parser.add_argument("--out_dir", default="results/dagig_rn03_10_grounded/grounding")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default=DINO_CONFIG)
    parser.add_argument("--weights", default=DINO_WEIGHTS)
    parser.add_argument("--max_rows", type=int, default=0)
    return parser.parse_args()


def score_key(row: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        float(row.get("iou_ge_0_3", 0.0)),
        float(row.get("center_hit_rate", 0.0)),
        float(row.get("detection_rate", 0.0)),
        -float(row.get("extreme_box_rate", 0.0)),
    )


def threshold_result(base: dict[str, object], box_threshold: float, text_threshold: float) -> dict[str, object]:
    preds = [
        pred
        for pred in base.get("all_predictions", [])
        if isinstance(pred, dict) and float(pred.get("score", 0.0) or 0.0) > box_threshold
    ]
    top = preds[0] if preds else None
    gold = base.get("gold_bbox_xyxy")
    pred_box = top.get("box_xyxy") if isinstance(top, dict) else None
    width = int(base.get("image_width", 1) or 1)
    height = int(base.get("image_height", 1) or 1)
    iou = bbox_iou(pred_box, gold)
    return {
        **base,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "pred_bbox_xyxy": pred_box,
        "best_score": top.get("score") if isinstance(top, dict) else None,
        "best_phrase": top.get("phrase") if isinstance(top, dict) else "",
        "num_detections": len(preds),
        "detected": pred_box is not None,
        "iou": iou,
        "iou_ge_0_1": iou >= 0.1,
        "iou_ge_0_3": iou >= 0.3,
        "iou_ge_0_5": iou >= 0.5,
        "center_hit": center_hit(pred_box, gold),
        "pred_gold_area_ratio": area_ratio(pred_box, gold),
        "extreme_box": is_extreme_box(pred_box, gold, width, height),
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    rows = load_jsonl(args.input_jsonl, limit=args.max_rows)
    if not rows:
        raise ValueError(f"No rows loaded from {args.input_jsonl}")
    model = load_groundingdino_model(args.config, args.weights, args.device)

    base_results = evaluate_grounding_rows(
        rows=rows,
        package_root=args.package_root,
        model=model,
        box_threshold=min(BOX_THRESHOLDS),
        text_threshold=min(TEXT_THRESHOLDS),
        device=args.device,
        out_dir=None,
        save_images=False,
    )
    grid_rows = []
    best: dict[str, float] | None = None
    for box_threshold in BOX_THRESHOLDS:
        for text_threshold in TEXT_THRESHOLDS:
            results = [threshold_result(row, box_threshold, text_threshold) for row in base_results]
            metrics = aggregate_grounding_metrics(results)
            row = {
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                **metrics,
            }
            grid_rows.append(row)
            if best is None or score_key(row) > score_key(best):
                best = row
            print(json.dumps(row, ensure_ascii=False))

    if best is None:
        raise RuntimeError("No threshold candidates evaluated")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "dev_threshold_grid.csv", grid_rows)
    write_json(out_dir / "best_threshold.json", best)
    print(f"best_threshold={json.dumps(best, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
