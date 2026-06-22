#!/usr/bin/env python3
"""Build autonomous full-image DAG-IG SFT data with a <locate> segment."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


DEFAULT_PACKAGE_NAME = "pix2fact_dagig_1k_gpt54_teacher_clean_package"
DEFAULT_MAIN_FILE = "data/pix2fact_dagig_train_AB_clean_split.jsonl"


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
    parser.add_argument("--out_dir", default="data/dagig_autonomous_sft")
    parser.add_argument("--main_file", default=DEFAULT_MAIN_FILE)
    parser.add_argument("--variant", default="autonomous_dagig_sft")
    parser.add_argument("--include_test", action="store_true")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path} n={len(rows)}")


def bbox_text(row: dict[str, Any]) -> str:
    teacher = row.get("gpt54_teacher")
    if isinstance(teacher, dict):
        rn_bbox = teacher.get("rn_locate_bbox_0_1000")
        if isinstance(rn_bbox, list) and len(rn_bbox) == 4:
            return json.dumps([int(round(float(v))) for v in rn_bbox])
    locate_target = row.get("locate_target")
    if isinstance(locate_target, str) and locate_target.strip():
        import re

        nums = re.findall(r"-?\d+(?:\.\d+)?", locate_target)
        if len(nums) >= 4:
            return json.dumps([int(round(float(v))) for v in nums[:4]])
    bbox = row.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Missing bbox_xyxy for sample_id={row.get('sample_id')}")
    values = [int(round(float(v))) for v in bbox]
    return json.dumps(values)


def prompt_text(row: dict[str, Any]) -> str:
    bbox_format = str(row.get("bbox_format") or "")
    if "0_1000" in bbox_format or "qwen" in bbox_format.lower():
        coord_text = "integer coordinates normalized to a 0-1000 image grid"
    else:
        coord_text = "pixel coordinates in the provided full image"
    return (
        "You are an autonomous multimodal search agent. Use only the full image and question. "
        "Return exactly these XML-like sections: <locate>, <observe>, <search_decision>, "
        f"<search>, <evidence>, <answer>. The <locate> section must be [x1, y1, x2, y2] "
        f"{coord_text}.\n\n"
        f"Question: {str(row.get('question', '')).strip()}"
    )


def make_example(chain_builder: Any, package_dir: Path, row: dict[str, Any], variant: str) -> dict[str, Any]:
    base = chain_builder.make_example(package_dir, row, "dagig_sft", min_process_weight=0.05)
    segments = dict(base["target_segments"])
    segments = {"locate": bbox_text(row), **segments}
    target = "\n".join(f"<{name}>\n{segments[name]}\n</{name}>" for name in ["locate", "observe", "search_decision", "search", "evidence", "answer"])
    locate_weight = float(base["segment_weights"].get("observe", base.get("training_weight", 1.0)))
    segment_weights = {"locate": locate_weight, **base["segment_weights"]}
    full_image = chain_builder.resolve_image_path(package_dir, row, "full_model_input")
    return {
        **base,
        "variant": variant,
        "task_type": "autonomous_full_image_chain",
        "input_mode": "full_image_only",
        "prompt": prompt_text(row),
        "target": target,
        "target_segments": segments,
        "segment_weights": segment_weights,
        "loss_weight": sum(float(v) for v in segment_weights.values()) / len(segment_weights),
        "messages": [
            {"role": "user", "content": prompt_text(row)},
            {"role": "assistant", "content": target},
        ],
        "images": [full_image],
        "full_image_path": full_image,
        "crop_image_path": "",
    }


def main() -> None:
    args = parse_args()
    chain_builder = load_chain_builder()
    package_dir = Path(args.package_dir).expanduser().resolve()
    main_path = Path(args.main_file)
    if not main_path.is_absolute():
        main_path = package_dir / main_path
    rows = chain_builder.load_jsonl(main_path)
    splits = {"train": [], "dev": [], "test": []}
    for row in rows:
        split = str(row.get("split", "train"))
        splits[split].append(make_example(chain_builder, package_dir, row, args.variant))
    out_dir = Path(args.out_dir).expanduser().resolve()
    write_jsonl(out_dir / f"{args.variant}_train.jsonl", splits["train"])
    write_jsonl(out_dir / f"{args.variant}_dev.jsonl", splits["dev"])
    if args.include_test:
        write_jsonl(out_dir / f"{args.variant}_test.jsonl", splits["test"])


if __name__ == "__main__":
    main()
