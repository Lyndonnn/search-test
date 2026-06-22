#!/usr/bin/env python3
"""Minimal GroundingDINO callable smoke test.

Loads a model checkpoint, runs one text-conditioned detection, and writes
machine-readable boxes plus a visual annotation/crop for quick inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from torchvision.ops import box_convert

from groundingdino.util.inference import annotate, load_image, load_model, predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument(
        "--weights",
        default="third_party/GroundingDINO_weights/groundingdino_swint_ogc.pth",
    )
    parser.add_argument("--image", default="third_party/GroundingDINO/.asset/cats.png")
    parser.add_argument("--text", default="cat")
    parser.add_argument("--out_dir", default="results/groundingdino_smoke")
    parser.add_argument("--box_threshold", type=float, default=0.30)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def clamp_box(xyxy: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def tensor_to_list(value: torch.Tensor) -> list[float]:
    return [float(v) for v in value.detach().cpu().tolist()]


def main() -> int:
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_source, image = load_image(args.image)
    height, width = image_source.shape[:2]

    model = load_model(args.config, args.weights, device=device)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=args.text,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device,
    )

    scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
    boxes_xyxy = box_convert(boxes=boxes * scale, in_fmt="cxcywh", out_fmt="xyxy")

    predictions: list[dict[str, Any]] = []
    for idx, (box_cxcywh, box_xyxy, score, phrase) in enumerate(
        zip(boxes, boxes_xyxy, logits, phrases)
    ):
        xyxy = tensor_to_list(box_xyxy)
        predictions.append(
            {
                "index": idx,
                "phrase": phrase,
                "score": float(score),
                "box_cxcywh_norm": tensor_to_list(box_cxcywh),
                "box_xyxy": xyxy,
                "box_xyxy_int": clamp_box(xyxy, width, height),
            }
        )

    payload = {
        "image": args.image,
        "caption": args.text,
        "config": args.config,
        "weights": args.weights,
        "device": device,
        "image_size": {"width": width, "height": height},
        "thresholds": {
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
        },
        "num_boxes": len(predictions),
        "predictions": predictions,
    }
    (out_dir / "predictions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if predictions:
        annotated = annotate(
            image_source=image_source,
            boxes=boxes,
            logits=logits,
            phrases=phrases,
        )
        cv2.imwrite(str(out_dir / "annotated.jpg"), annotated)

        x1, y1, x2, y2 = predictions[0]["box_xyxy_int"]
        with Image.open(args.image).convert("RGB") as pil_image:
            pil_image.crop((x1, y1, x2, y2)).save(out_dir / "crop_00.jpg")

    summary = {
        "ok": bool(predictions),
        "num_boxes": len(predictions),
        "top_prediction": predictions[0] if predictions else None,
        "out_dir": str(out_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if predictions else 2


if __name__ == "__main__":
    raise SystemExit(main())
