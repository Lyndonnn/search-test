#!/usr/bin/env python3
"""Build grounded DAG-IG RL rows from ground-action SFT data and DINO results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from grounded_pipeline_utils import (
    evidence_text,
    extract_segments,
    has_forbidden_image_term,
    load_jsonl,
    md_table,
    write_json,
    write_jsonl,
)


TAGS = ["ground", "observe", "search_decision", "search", "evidence", "answer"]
DAG_GRAPH = {
    "nodes": [
        "image_question",
        "ground_expression",
        "grounding_tool_bbox_crop",
        "observation",
        "search_query",
        "retrieved_evidence",
        "answer",
    ],
    "edges": [
        ["image_question", "ground_expression"],
        ["ground_expression", "grounding_tool_bbox_crop"],
        ["grounding_tool_bbox_crop", "observation"],
        ["observation", "search_query"],
        ["search_query", "retrieved_evidence"],
        ["retrieved_evidence", "answer"],
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", default="data/dagig_rn03_10_grounded/ground_action_train.jsonl")
    parser.add_argument("--dev_file", default="data/dagig_rn03_10_grounded/ground_action_dev.jsonl")
    parser.add_argument("--test_file", default="data/dagig_rn03_10_grounded/ground_action_test.jsonl")
    parser.add_argument(
        "--grounding_train",
        default="results/dagig_rn03_10_grounded/grounding/final_train/grounding_results.jsonl",
    )
    parser.add_argument(
        "--grounding_dev",
        default="results/dagig_rn03_10_grounded/grounding/final_dev/grounding_results.jsonl",
    )
    parser.add_argument(
        "--grounding_test",
        default="results/dagig_rn03_10_grounded/grounding/final_test/grounding_results.jsonl",
    )
    parser.add_argument("--out_dir", default="data/dagig_rn03_10_grounded_rl")
    return parser.parse_args()


def first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def grounding_by_sample(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError(f"Grounding row without sample_id in {path}")
        if sample_id in out:
            raise ValueError(f"Duplicate grounding sample_id={sample_id} in {path}")
        out[sample_id] = row
    return out


def teacher_segments(row: dict[str, Any]) -> dict[str, str]:
    target = row.get("target_segments")
    if isinstance(target, dict):
        return {tag: str(target.get(tag) or "").strip() for tag in TAGS}
    return extract_segments(str(row.get("model_target_text") or row.get("target") or ""), TAGS)


def build_row(row: dict[str, Any], dino: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError("SFT row without sample_id")
    image_path = first_text(row.get("clean_rn_image_path"), row.get("full_image_path"), *(row.get("images") or []))
    image_relpath = first_text(row.get("clean_rn_image_relpath_from_package"), dino.get("image_relpath"))
    if not image_path:
        raise ValueError(f"sample_id={sample_id} missing clean image path")
    if has_forbidden_image_term(image_path) or has_forbidden_image_term(image_relpath):
        raise ValueError(f"sample_id={sample_id} appears to use non-clean image path: {image_path}")
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"sample_id={sample_id} image missing: {image_path}")

    segments = teacher_segments(row)
    teacher_evidence = first_text(segments.get("evidence"), evidence_text(row))
    result = {
        "detection": bool(dino.get("detected")),
        "best_bbox_pixel_xyxy": dino.get("pred_bbox_xyxy"),
        "score": dino.get("best_score"),
        "phrase": dino.get("best_phrase"),
        "iou": dino.get("iou"),
        "center_hit": bool(dino.get("center_hit")),
        "extreme_box": bool(dino.get("extreme_box")),
        "num_detections": int(dino.get("num_detections") or 0),
        "crop_path": str(dino.get("crop_path") or ""),
        "vis_path": str(dino.get("vis_path") or ""),
        "gold_bbox_pixel_xyxy": dino.get("gold_bbox_xyxy"),
    }

    return {
        "sample_id": sample_id,
        "split": str(row.get("split") or dino.get("split") or ""),
        "clean_rn_image_path": image_path,
        "clean_rn_image_relpath_from_package": image_relpath,
        "images": [image_path],
        "question": str(row.get("question") or ""),
        "gold_answer": row.get("answer"),
        "answer": row.get("answer"),
        "teacher_ground_expression": first_text(row.get("ground_expression"), segments.get("ground")),
        "ground_expression": first_text(row.get("ground_expression"), segments.get("ground")),
        "teacher_observation": first_text(segments.get("observe")),
        "teacher_search_query": first_text(segments.get("search")),
        "teacher_evidence_text": teacher_evidence,
        "evidence": teacher_evidence,
        "evidences": row.get("evidences") if isinstance(row.get("evidences"), list) else [],
        "evidence_support": row.get("evidence_support"),
        "semantic_anchor": str(row.get("semantic_anchor") or ""),
        "visual_anchor": str(row.get("visual_anchor") or ""),
        "groundingdino_teacher_result": result,
        "original_model_target_text": str(row.get("model_target_text") or row.get("target") or ""),
        "model_target_text": str(row.get("model_target_text") or row.get("target") or ""),
        "target_segments": segments,
        "ground_tool_gold": row.get("ground_tool_gold"),
        "dag_graph": DAG_GRAPH,
        "dagig_ground_graph": row.get("dagig_ground_graph") or DAG_GRAPH,
        "dagig_reward_targets": row.get("dagig_reward_targets") or {},
        "tool_result_grounding": row.get("tool_result_grounding") or {},
        "prompt": str(row.get("prompt") or ""),
        "messages": row.get("messages") if isinstance(row.get("messages"), list) else [],
        "loss_weight": row.get("loss_weight", 1.0),
        "segment_weights": row.get("segment_weights") if isinstance(row.get("segment_weights"), dict) else {},
        "task_type": "grounded_dagig_rl",
        "input_mode": "clean_rn_image_only",
        "leakage_sensitive_exclude": False,
    }


def build_split(split: str, sft_path: str, grounding_path: str, out_dir: Path) -> dict[str, Any]:
    sft_rows = load_jsonl(sft_path)
    dino_rows = grounding_by_sample(grounding_path)
    out_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in sft_rows:
        sample_id = str(row.get("sample_id") or "")
        dino = dino_rows.get(sample_id)
        if dino is None:
            missing.append(sample_id)
            continue
        out_rows.append(build_row(row, dino))
    if missing:
        raise ValueError(f"{split}: missing GroundingDINO rows for {len(missing)} samples, first={missing[:5]}")
    out_path = out_dir / f"grounded_rl_{split}.jsonl"
    write_jsonl(out_path, out_rows)
    return {
        "split": split,
        "sft_rows": len(sft_rows),
        "grounding_rows": len(dino_rows),
        "written": len(out_rows),
        "output": str(out_path),
        "detections": sum(1 for row in out_rows if row["groundingdino_teacher_result"]["detection"]),
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        build_split("train", args.train_file, args.grounding_train, out_dir),
        build_split("dev", args.dev_file, args.grounding_dev, out_dir),
        build_split("test", args.test_file, args.grounding_test, out_dir),
    ]
    expected = {"train": 458, "dev": 98, "test": 64}
    failures = [
        f"{row['split']} expected {expected[row['split']]} wrote {row['written']}"
        for row in summaries
        if row["written"] != expected[row["split"]]
    ]
    summary = {
        "ok": not failures,
        "expected_counts": expected,
        "splits": summaries,
        "failures": failures,
    }
    write_json(out_dir / "grounded_rl_data_summary.json", summary)
    metric_rows = [{k: v for k, v in row.items() if k != "output"} for row in summaries]
    md = "# Grounded RL Data Summary\n\n" + md_table(metric_rows) + "\n"
    if failures:
        md += "\n## Failures\n\n" + "\n".join(f"- {item}" for item in failures) + "\n"
    (out_dir / "grounded_rl_data_summary.md").write_text(md, encoding="utf-8")
    print(md)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
