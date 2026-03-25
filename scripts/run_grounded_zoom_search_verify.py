#!/usr/bin/env python3
import argparse
import json
import os
import sys
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the grounded_zoom_search_verify inference prototype.")
    parser.add_argument("--agent-mode", default="grounded_zoom_search_verify", help="Experimental agent mode name.")
    parser.add_argument("--model-path", required=True, help="HF model path.")
    parser.add_argument("--image", default="", help="Single image path or URL.")
    parser.add_argument("--question", default="", help="Single question.")
    parser.add_argument("--parquet", default="", help="Optional parquet path for sample-based execution.")
    parser.add_argument("--index", type=int, default=0, help="Row index when --parquet is provided.")
    parser.add_argument("--image-dir", default="", help="Optional image directory fallback joined with original_image_name.")
    parser.add_argument("--output-json", required=True, help="Where to save the final JSON result.")
    parser.add_argument("--trace-jsonl", required=True, help="Where to save action traces as JSONL.")
    parser.add_argument("--workdir", default="", help="Optional directory to store region crops.")
    parser.add_argument("--grid-sizes", default="1,2", help="Comma-separated grid sizes, e.g. 1,2 or 1,2,3")
    parser.add_argument("--topk-regions", type=int, default=3)
    parser.add_argument("--max-zoom-steps", type=int, default=1)
    parser.add_argument("--bbox-padding", type=float, default=0.05)
    parser.add_argument("--disable-ocr", action="store_true")
    parser.add_argument("--disable-caption", action="store_true")
    parser.add_argument("--disable-image-search", action="store_true")
    parser.add_argument("--disable-text-search", action="store_true")
    parser.add_argument("--image-search-limit", type=int, default=1)
    parser.add_argument("--text-search-limit", type=int, default=1)
    parser.add_argument("--search-parquet", default="", help="Optional offline search parquet.")
    return parser.parse_args()


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


def resolve_row_image_path(row: pd.Series, image_dir: str) -> str:
    candidates = [
        safe_text(row.get("image_path_local", "")),
        safe_text(row.get("image_path_drive", "")),
    ]
    original_name = safe_text(row.get("original_image_name", ""))
    if image_dir and original_name:
        candidates.append(os.path.join(image_dir, original_name))
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No image path exists for row {safe_text(row.get('qid', ''))}: {candidates}")


def main() -> None:
    args = parse_args()
    if args.search_parquet:
        os.environ["MMSEARCH_OFFLINE_PARQUET"] = args.search_parquet

    from mmsearch_r1.agents.grounded_zoom_search_verify import (
        GroundedZoomSearchVerifyAgent,
        GroundedZoomSearchVerifyConfig,
    )
    from mmsearch_r1.scripts.inference_torch_demo import load_model_and_processor

    if args.parquet:
        df = pd.read_parquet(args.parquet)
        row = df.iloc[args.index]
        image_path = resolve_row_image_path(row, args.image_dir)
        question = safe_text(row.get("question", ""))
    else:
        if not args.image or not args.question:
            raise ValueError("Provide either --parquet/--index or both --image and --question.")
        image_path = args.image
        question = args.question

    grid_sizes = tuple(int(part.strip()) for part in args.grid_sizes.split(",") if part.strip())
    config = GroundedZoomSearchVerifyConfig(
        grid_sizes=grid_sizes,
        topk_regions=args.topk_regions,
        max_zoom_steps=args.max_zoom_steps,
        bbox_padding=args.bbox_padding,
        enable_ocr=not args.disable_ocr,
        enable_caption=not args.disable_caption,
        enable_image_search=not args.disable_image_search,
        enable_text_search=not args.disable_text_search,
        image_search_limit=args.image_search_limit,
        text_search_limit=args.text_search_limit,
    )

    model, processor = load_model_and_processor(args.model_path)
    agent = GroundedZoomSearchVerifyAgent(model, processor, config=config)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.trace_jsonl) or ".", exist_ok=True)

    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)
        result = agent.run(image_path, question, args.workdir)
    else:
        with TemporaryDirectory(prefix="gzsv_") as tmpdir:
            result = agent.run(image_path, question, tmpdir)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(args.trace_jsonl, "w", encoding="utf-8") as f:
        for event in result["trace_events"]:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    print(f"Saved result JSON to {args.output_json}")
    print(f"Saved trace JSONL to {args.trace_jsonl}")
    print(json.dumps(result["final_answer"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
