#!/usr/bin/env python3
"""Create contact sheets for teacher-expression GroundingDINO results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from grounded_pipeline_utils import draw_boxes, load_jsonl, md_table, select_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev_results", default="results/dagig_rn03_10_grounded/grounding/final_dev/grounding_results.jsonl")
    parser.add_argument("--test_results", default="results/dagig_rn03_10_grounded/grounding/final_test/grounding_results.jsonl")
    parser.add_argument("--out_dir", default="results/dagig_rn03_10_grounded/contact_sheets")
    parser.add_argument("--per_sheet", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def resize_for_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    copy = image.copy()
    copy.thumbnail((size[0], size[1] - 96))
    x = (size[0] - copy.width) // 2
    y = 0
    canvas.paste(copy, (x, y))
    return canvas


def wrap(text: Any, max_chars: int) -> list[str]:
    words = str(text or "").replace("\n", " ").split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > max_chars and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines[:4]


def make_panel(row: dict[str, Any], panel_size: tuple[int, int]) -> Image.Image:
    image = draw_boxes(
        row["image_path"],
        gold=row.get("gold_bbox_xyxy"),
        pred=row.get("pred_bbox_xyxy"),
    )
    panel = resize_for_panel(image, panel_size)
    draw = ImageDraw.Draw(panel)
    y = panel_size[1] - 92
    score = row.get("best_score")
    score_text = "NA" if score is None else f"{float(score):.3f}"
    lines = [
        f"{row.get('sample_id')} {row.get('question_type')} score={score_text} IoU={float(row.get('iou', 0.0)):.3f} hit={row.get('center_hit')}",
        f"expr: {row.get('ground_expression')}",
        f"answer: {row.get('answer')}",
    ]
    for raw in lines:
        for line in wrap(raw, 54):
            draw.text((6, y), line, fill=(0, 0, 0))
            y += 12
            if y > panel_size[1] - 4:
                return panel
    return panel


def make_sheet(name: str, rows: list[dict[str, Any]], out_dir: Path, per_sheet: int) -> dict[str, Any]:
    rows = rows[:per_sheet]
    panel_size = (360, 360)
    cols = 4
    sheet_rows = max(1, (len(rows) + cols - 1) // cols)
    sheet = Image.new("RGB", (panel_size[0] * cols, panel_size[1] * sheet_rows), "white")
    draw = ImageDraw.Draw(sheet)
    if not rows:
        draw.text((20, 20), f"{name}: no matching rows", fill=(0, 0, 0))
    for idx, row in enumerate(rows):
        panel = make_panel(row, panel_size)
        x = (idx % cols) * panel_size[0]
        y = (idx // cols) * panel_size[1]
        sheet.paste(panel, (x, y))
    out_path = out_dir / f"{name}.jpg"
    sheet.save(out_path, quality=92)
    return {"sheet": name, "path": str(out_path), "n": len(rows)}


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = load_jsonl(args.dev_results)
    test = load_jsonl(args.test_results)

    selections = {
        "random_dev": select_rows(dev, args.per_sheet, args.seed),
        "worst_dev_by_iou": sorted(dev, key=lambda r: float(r.get("iou", 0.0)))[: args.per_sheet],
        "no_detection_dev": [r for r in dev if not r.get("detected")][: args.per_sheet],
        "high_score_wrong_dev": sorted(
            [r for r in dev if r.get("detected") and float(r.get("iou", 0.0)) < 0.1],
            key=lambda r: float(r.get("best_score", 0.0) or 0.0),
            reverse=True,
        )[: args.per_sheet],
        "good_dev": sorted(
            [r for r in dev if float(r.get("iou", 0.0)) >= 0.3 and r.get("center_hit")],
            key=lambda r: float(r.get("iou", 0.0)),
            reverse=True,
        )[: args.per_sheet],
        "random_test": select_rows(test, args.per_sheet, args.seed + 1),
        "worst_test": sorted(test, key=lambda r: float(r.get("iou", 0.0)))[: args.per_sheet],
    }
    summary = [make_sheet(name, rows, out_dir, args.per_sheet) for name, rows in selections.items()]
    (out_dir / "contact_sheet_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "contact_sheet_summary.md").write_text("# Contact Sheets\n\n" + md_table(summary) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

