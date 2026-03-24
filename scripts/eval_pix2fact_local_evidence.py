#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import sys
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmsearch_r1.utils.reward_score_mm.mmsearch_r1_score import em_check, extract_solution, subem_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whole-image vs oracle-crop baselines on Pix2Fact-style clean parquet."
    )
    parser.add_argument("--parquet", required=True, help="Path to Pix2Fact clean parquet.")
    parser.add_argument("--model-path", required=True, help="HF model path, e.g. lmms-lab/MMSearch-R1-7B")
    parser.add_argument("--output", required=True, help="Where to save JSON results.")
    parser.add_argument(
        "--mode",
        choices=["whole", "oracle_crop", "both"],
        default="both",
        help="Evaluation mode. 'both' is recommended for methodology validation.",
    )
    parser.add_argument(
        "--search-parquet",
        default="",
        help="Optional veRL parquet used as the offline search corpus.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Number of rows to evaluate.")
    parser.add_argument("--offset", type=int, default=0, help="Start row offset.")
    parser.add_argument(
        "--bbox-format",
        choices=["auto", "xyxy", "xywh"],
        default="auto",
        help="Interpretation of 4-number bounding boxes.",
    )
    parser.add_argument(
        "--bbox-padding",
        type=float,
        default=0.05,
        help="Padding ratio applied to the parsed bbox before cropping.",
    )
    parser.add_argument(
        "--image-dir",
        default="",
        help="Optional fallback directory joined with original_image_name when parquet paths are stale.",
    )
    parser.add_argument(
        "--save-crops-dir",
        default="",
        help="Optional directory to save oracle crops for inspection.",
    )
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


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if hasattr(value, "item"):
        return to_jsonable(value.item())
    return str(value)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if hasattr(value, "item"):
        try:
            return safe_text(value.item())
        except Exception:
            pass
    return str(value).strip()


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def resolve_image_path(row: pd.Series, image_dir: str) -> str:
    candidates = [
        safe_text(row.get("image_path_local", "")),
        safe_text(row.get("image_path_drive", "")),
    ]
    original_image_name = safe_text(row.get("original_image_name", ""))
    if image_dir and original_image_name:
        candidates.append(os.path.join(image_dir, original_image_name))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No image path exists for qid={safe_text(row.get('qid', ''))}: {candidates}")


def _extract_numeric_list(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        numbers: list[float] = []
        for item in value:
            if isinstance(item, (int, float)):
                numbers.append(float(item))
        return numbers
    text = safe_text(value)
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, dict):
                return []
            if isinstance(parsed, (list, tuple)):
                return _extract_numeric_list(parsed)
        except Exception:
            continue
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return [float(m) for m in matches]


def _bbox_from_region_dict(region: Any, image_size: tuple[int, int]) -> Optional[List[int]]:
    if not isinstance(region, dict):
        return None

    point_candidates = []
    for key in ("tl", "tr", "bl", "br", "lt", "rt", "lb", "rb"):
        point = region.get(key)
        if isinstance(point, dict) and "x" in point and "y" in point:
            try:
                point_candidates.append((float(point["x"]), float(point["y"])))
            except Exception:
                pass

    if point_candidates:
        xs = [point[0] for point in point_candidates]
        ys = [point[1] for point in point_candidates]
        return finalize_bbox([min(xs), min(ys), max(xs), max(ys)], image_size, "xyxy")

    if {"x", "y", "width", "height"}.issubset(region):
        return finalize_bbox(
            [region["x"], region["y"], region["width"], region["height"]],
            image_size,
            "xywh",
        )
    if {"x", "y", "w", "h"}.issubset(region):
        return finalize_bbox([region["x"], region["y"], region["w"], region["h"]], image_size, "xywh")
    if {"left", "top", "right", "bottom"}.issubset(region):
        return finalize_bbox(
            [region["left"], region["top"], region["right"], region["bottom"]],
            image_size,
            "xyxy",
        )
    return None


def parse_bbox(value: Any, image_size: tuple[int, int], bbox_format: str) -> Optional[List[int]]:
    width, height = image_size

    parsed_value = value
    if isinstance(value, str):
        text = safe_text(value)
        if not text:
            return None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed_value = parser(text)
                break
            except Exception:
                parsed_value = text

    if isinstance(parsed_value, dict):
        keys = {k.lower(): v for k, v in parsed_value.items()}
        region_box = _bbox_from_region_dict(keys.get("region"), image_size)
        if region_box is not None:
            return region_box
        if {"x1", "y1", "x2", "y2"}.issubset(keys):
            coords = [keys["x1"], keys["y1"], keys["x2"], keys["y2"]]
            return finalize_bbox(coords, image_size, "xyxy")
        if {"x", "y", "w", "h"}.issubset(keys):
            coords = [keys["x"], keys["y"], keys["w"], keys["h"]]
            return finalize_bbox(coords, image_size, "xywh")
        if {"left", "top", "right", "bottom"}.issubset(keys):
            coords = [keys["left"], keys["top"], keys["right"], keys["bottom"]]
            return finalize_bbox(coords, image_size, "xyxy")
        return None

    if isinstance(parsed_value, (list, tuple)):
        if len(parsed_value) == 1 and isinstance(parsed_value[0], (list, tuple, dict, str)):
            return parse_bbox(parsed_value[0], image_size, bbox_format)
        if len(parsed_value) >= 4 and all(isinstance(v, (int, float)) for v in parsed_value[:4]):
            fmt = bbox_format
            if fmt == "auto":
                x1, y1, x3, y4 = [float(v) for v in parsed_value[:4]]
                fmt = "xyxy" if x3 > x1 and y4 > y1 else "xywh"
            return finalize_bbox(parsed_value[:4], image_size, fmt)

    coords = _extract_numeric_list(parsed_value)
    if len(coords) >= 4:
        fmt = bbox_format
        if fmt == "auto":
            fmt = "xyxy" if coords[2] > coords[0] and coords[3] > coords[1] else "xywh"
        return finalize_bbox(coords[:4], image_size, fmt)
    return None


def finalize_bbox(coords: Any, image_size: tuple[int, int], bbox_format: str) -> Optional[List[int]]:
    width, height = image_size
    x1, y1, a3, a4 = [float(v) for v in coords[:4]]

    # Normalized boxes.
    if max(abs(x1), abs(y1), abs(a3), abs(a4)) <= 1.5:
        x1 *= width
        y1 *= height
        if bbox_format == "xywh":
            a3 *= width
            a4 *= height
        else:
            a3 *= width
            a4 *= height

    if bbox_format == "xywh":
        x2 = x1 + a3
        y2 = y1 + a4
    else:
        x2 = a3
        y2 = a4

    x1_i = max(0, min(width - 1, int(round(x1))))
    y1_i = max(0, min(height - 1, int(round(y1))))
    x2_i = max(1, min(width, int(round(x2))))
    y2_i = max(1, min(height, int(round(y2))))

    if x2_i <= x1_i or y2_i <= y1_i:
        return None
    return [x1_i, y1_i, x2_i, y2_i]


def apply_padding(box: List[int], image_size: tuple[int, int], padding_ratio: float) -> List[int]:
    if padding_ratio <= 0:
        return box
    width, height = image_size
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = int(round(box_w * padding_ratio))
    pad_h = int(round(box_h * padding_ratio))
    return [
        max(0, x1 - pad_w),
        max(0, y1 - pad_h),
        min(width, x2 + pad_w),
        min(height, y2 + pad_h),
    ]


def compute_metrics(trajectory: Dict[str, Any], answers: List[str]) -> Dict[str, Any]:
    final_answer = extract_solution(trajectory["final_response"])
    em = bool(final_answer is not None and em_check(final_answer, answers))
    subem = bool(final_answer is not None and subem_check(final_answer, answers))
    search_count = sum(
        int("<search><img></search>" in resp) + int("<text_search>" in resp and "</text_search>" in resp)
        for resp in trajectory["responses"]
    )
    return {
        "final_answer": final_answer,
        "em": em,
        "subem": subem,
        "search_count": search_count,
        "responses": trajectory["responses"],
        "tool_trace": trajectory["tool_trace"],
    }


def summarize(results: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    eligible = [r[key] for r in results if key in r and r[key].get("status") == "ok"]
    if not eligible:
        return {"count": 0, "em": 0.0, "subem": 0.0, "avg_searches": 0.0}
    count = len(eligible)
    return {
        "count": count,
        "em": sum(int(r["em"]) for r in eligible) / count,
        "subem": sum(int(r["subem"]) for r in eligible) / count,
        "avg_searches": sum(int(r["search_count"]) for r in eligible) / count,
    }


def main() -> None:
    args = parse_args()
    if args.search_parquet:
        os.environ["MMSEARCH_OFFLINE_PARQUET"] = args.search_parquet

    from mmsearch_r1.scripts.inference_torch_demo import load_model_and_processor, run_mmsearch_demo

    df = pd.read_parquet(args.parquet)
    end = min(len(df), args.offset + args.limit)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.save_crops_dir:
        os.makedirs(args.save_crops_dir, exist_ok=True)

    model, processor = load_model_and_processor(args.model_path)
    rows: List[Dict[str, Any]] = []

    with TemporaryDirectory(prefix="pix2fact_oracle_") as tmpdir:
        for row_idx in range(args.offset, end):
            row = df.iloc[row_idx]
            qid = safe_text(row.get("qid", f"row_{row_idx}"))
            question = safe_text(row.get("question", ""))
            answer = safe_text(row.get("answer", ""))
            aliases = [safe_text(v) for v in as_sequence(row.get("aliases", [])) if safe_text(v)]
            answers = [answer] + [a for a in aliases if a and a != answer]
            image_path = resolve_image_path(row, args.image_dir)

            sample_result: Dict[str, Any] = {
                "row_index": row_idx,
                "qid": qid,
                "item_id": safe_text(row.get("item_id", "")),
                "question": question,
                "answer": answer,
                "aliases": aliases,
                "original_image_name": safe_text(row.get("original_image_name", "")),
                "image_path": image_path,
                "bounding_box_raw": to_jsonable(row.get("bounding_box", "")),
            }

            if args.mode in {"whole", "both"}:
                trajectory = run_mmsearch_demo(model, processor, image_path, question)
                sample_result["whole"] = {"status": "ok", **compute_metrics(trajectory, answers)}

            if args.mode in {"oracle_crop", "both"}:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    crop_box = parse_bbox(row.get("bounding_box", ""), image.size, args.bbox_format)
                    if crop_box is None:
                        sample_result["oracle_crop"] = {"status": "missing_bbox"}
                    else:
                        crop_box = apply_padding(crop_box, image.size, args.bbox_padding)
                        crop = image.crop(tuple(crop_box))
                        crop_dir = args.save_crops_dir or tmpdir
                        crop_path = os.path.join(crop_dir, f"{qid}_crop.png")
                        crop.save(crop_path)
                        crop_ratio = ((crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])) / (
                            image.size[0] * image.size[1]
                        )
                        trajectory = run_mmsearch_demo(model, processor, crop_path, question)
                        sample_result["oracle_crop"] = {
                            "status": "ok",
                            "crop_box": crop_box,
                            "crop_area_ratio": crop_ratio,
                            "crop_path": crop_path,
                            **compute_metrics(trajectory, answers),
                        }

            rows.append(to_jsonable(sample_result))

            whole_msg = ""
            if "whole" in sample_result:
                whole = sample_result["whole"]
                whole_msg = f" whole(em={int(whole['em'])},subem={int(whole['subem'])},searches={whole['search_count']})"
            crop_msg = ""
            if "oracle_crop" in sample_result:
                crop = sample_result["oracle_crop"]
                if crop["status"] == "ok":
                    crop_msg = f" crop(em={int(crop['em'])},subem={int(crop['subem'])},searches={crop['search_count']})"
                else:
                    crop_msg = " crop(missing_bbox)"
            print(f"[{ordinal(len(rows))}/{end - args.offset}] qid={qid}{whole_msg}{crop_msg}")

    summary = {}
    if args.mode in {"whole", "both"}:
        summary["whole"] = summarize(rows, "whole")
    if args.mode in {"oracle_crop", "both"}:
        summary["oracle_crop"] = summarize(rows, "oracle_crop")
    if args.mode == "both":
        paired = [
            row for row in rows
            if row.get("whole", {}).get("status") == "ok" and row.get("oracle_crop", {}).get("status") == "ok"
        ]
        if paired:
            summary["paired_delta"] = {
                "count": len(paired),
                "crop_minus_whole_em": (
                    sum(int(row["oracle_crop"]["em"]) - int(row["whole"]["em"]) for row in paired) / len(paired)
                ),
                "crop_minus_whole_subem": (
                    sum(int(row["oracle_crop"]["subem"]) - int(row["whole"]["subem"]) for row in paired) / len(paired)
                ),
            }

    payload = {"config": vars(args), "summary": summary, "rows": rows}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {args.output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
