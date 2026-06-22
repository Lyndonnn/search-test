#!/usr/bin/env python3
"""Evaluate teacher ground expressions with local GroundingDINO."""

from __future__ import annotations

import argparse
from pathlib import Path

from grounded_pipeline_utils import (
    DINO_CONFIG,
    DINO_WEIGHTS,
    aggregate_grounding_metrics,
    evaluate_grounding_rows,
    load_groundingdino_model,
    load_jsonl,
    write_grounding_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--package_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--config", default=DINO_CONFIG)
    parser.add_argument("--weights", default=DINO_WEIGHTS)
    parser.add_argument("--no_save_images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.input_jsonl, limit=args.max_rows)
    if not rows:
        raise ValueError(f"No rows loaded from {args.input_jsonl}")
    model = load_groundingdino_model(args.config, args.weights, args.device)
    results = evaluate_grounding_rows(
        rows=rows,
        package_root=args.package_root,
        model=model,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
        out_dir=args.out_dir,
        save_images=not args.no_save_images,
    )
    metrics = aggregate_grounding_metrics(results)
    metrics.update(
        {
            "input_jsonl": args.input_jsonl,
            "package_root": str(Path(args.package_root)),
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "device": args.device,
        }
    )
    write_grounding_outputs(args.out_dir, results, metrics)
    print(metrics)
    return 0 if metrics.get("detection_rate", 0.0) > 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())

