#!/usr/bin/env python3
"""Build ground-action SFT data from hard-pass rows and GroundingDINO results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from grounded_pipeline_utils import (
    HARD_FILES,
    PACKAGE_DIR,
    evidence_text,
    hard_pass,
    load_jsonl,
    row_image_path,
    write_jsonl,
)


SEGMENTS = ["ground", "observe", "search_decision", "search", "evidence", "answer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package_root", default=str(PACKAGE_DIR))
    parser.add_argument("--grounding_train", default="results/dagig_rn03_10_grounded/grounding/final_train/grounding_results.jsonl")
    parser.add_argument("--grounding_dev", default="results/dagig_rn03_10_grounded/grounding/final_dev/grounding_results.jsonl")
    parser.add_argument("--grounding_test", default="results/dagig_rn03_10_grounded/grounding/final_test/grounding_results.jsonl")
    parser.add_argument("--out_dir", default="data/dagig_rn03_10_grounded")
    return parser.parse_args()


def teacher(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("gpt54_teacher")
    return value if isinstance(value, dict) else {}


def bool_text(value: Any) -> str:
    return "search" if bool(value) else "answer"


def search_query(row: dict[str, Any]) -> str:
    t = teacher(row)
    for value in (t.get("repaired_search_query"), t.get("search_query"), row.get("hf_search_query")):
        text = str(value or "").strip()
        if text:
            return text
    return str(row.get("semantic_anchor") or row.get("ground_expression") or "").strip()


def sanitize_observation(text: str, answer: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"\b[Tt]he crop shows\b", "The grounded region shows", value)
    value = re.sub(r"\b[Cc]rop shows\b", "the grounded region shows", value)
    value = re.sub(r"\bcrop\b", "grounded region", value, flags=re.I)
    value = re.sub(r"\bred[- ]?box(?:ed)?\b", "grounded", value, flags=re.I)
    value = re.sub(r"\bbbox\b|\bbounding box\b|\bannotation(?:s|ed)?\b", "grounding cue", value, flags=re.I)
    ans = str(answer or "").strip()
    if ans and len(ans) >= 2:
        value = re.sub(re.escape(ans), "the target answer", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip()


def observation(row: dict[str, Any]) -> str:
    t = teacher(row)
    text = t.get("local_observation") or row.get("new_local_observation") or ""
    if not text:
        for item in row.get("trajectory_ground_action_v3") or row.get("trajectory") or []:
            if isinstance(item, dict) and item.get("action") == "observe_crop":
                text = item.get("output") or ""
                break
    return sanitize_observation(str(text), str(row.get("answer") or ""))


def action_rewards(row: dict[str, Any]) -> dict[str, float]:
    rewards = teacher(row).get("action_rewards")
    if not isinstance(rewards, dict):
        rewards = (row.get("reward_variants") or {}).get("dagig") if isinstance(row.get("reward_variants"), dict) else {}
    if not isinstance(rewards, dict):
        rewards = {}

    def get(key: str, default: float = 1.0) -> float:
        try:
            return max(0.0, min(1.0, float(rewards.get(key, default))))
        except (TypeError, ValueError):
            return default

    return {
        "ground": get("observe_crop", 1.0),
        "observe": get("observe_crop", 1.0),
        "search_decision": get("search_query", 1.0),
        "search": get("search_query", 1.0),
        "evidence": get("evidence_selection", 1.0),
        "answer": get("answer", 1.0),
    }


def format_target(segments: dict[str, str]) -> str:
    return "\n".join(f"<{name}>{segments[name]}</{name}>" for name in SEGMENTS)


def prompt_text(row: dict[str, Any]) -> str:
    return (
        "You are a multimodal search agent. Use the clean region-normalized image and the question. "
        "First write a perceptual grounding expression for the visual target, then continue the search trajectory. "
        "Return exactly these XML-like sections: <ground>, <observe>, <search_decision>, <search>, <evidence>, <answer>. "
        "Do not output bounding boxes or tool-result fields.\n\n"
        f"Question: {str(row.get('question', '')).strip()}"
    )


def tool_result(row: dict[str, Any], grounding_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "environment_observation_not_model_target",
        "source": "groundingdino_teacher_ground_expression",
        "ground_expression": row.get("ground_expression"),
        "pred_bbox_pixel_xyxy": grounding_result.get("pred_bbox_xyxy"),
        "pred_score": grounding_result.get("best_score"),
        "pred_phrase": grounding_result.get("best_phrase"),
        "gold_bbox_pixel_xyxy": (row.get("ground_tool_gold") or {}).get("bbox_pixel_xyxy"),
        "iou": grounding_result.get("iou"),
        "center_hit": grounding_result.get("center_hit"),
    }


def make_example(package_root: Path, row: dict[str, Any], grounding_result: dict[str, Any]) -> dict[str, Any]:
    if not hard_pass(row):
        raise ValueError(f"review/non-hard row passed into SFT build: {row.get('sample_id')}")
    image_path = str(row_image_path(package_root, row).resolve())
    segments = {
        "ground": str(row.get("ground_expression") or "").strip(),
        "observe": observation(row),
        "search_decision": "search",
        "search": search_query(row),
        "evidence": evidence_text(row),
        "answer": str(row.get("answer") or "").strip(),
    }
    if any(not segments[name] for name in SEGMENTS):
        raise ValueError(f"empty target segment for sample_id={row.get('sample_id')}: {segments}")
    target = format_target(segments)
    weights = action_rewards(row)
    return {
        "sample_id": row.get("sample_id"),
        "split": row.get("split"),
        "variant": "ground_action_sft",
        "task_type": "ground_action_full_clean_rn_chain",
        "input_mode": "clean_rn_image_only",
        "question": row.get("question"),
        "answer": row.get("answer"),
        "clean_rn_image_path": image_path,
        "clean_rn_image_relpath_from_package": row.get("clean_rn_image_relpath_from_package"),
        "ground_expression": segments["ground"],
        "semantic_anchor": row.get("semantic_anchor") or (row.get("grounding") or {}).get("semantic_anchor"),
        "model_target_text": target,
        "tool_result_grounding": tool_result(row, grounding_result),
        "ground_tool_gold": row.get("ground_tool_gold"),
        "dagig_ground_graph": {
            "nodes": [
                {"id": "question", "type": "question", "text": row.get("question")},
                {"id": "ground_expression", "type": "model_action", "text": segments["ground"]},
                {"id": "grounding_tool", "type": "environment", "text": "GroundingDINO bbox/crop observation"},
                {"id": "observe", "type": "model_action", "text": segments["observe"]},
                {"id": "search", "type": "model_action", "text": segments["search"]},
                {"id": "evidence", "type": "environment", "text": segments["evidence"]},
                {"id": "answer", "type": "model_action", "text": segments["answer"]},
            ],
            "edges": [
                ["question", "ground_expression"],
                ["ground_expression", "grounding_tool"],
                ["grounding_tool", "observe"],
                ["observe", "search"],
                ["search", "evidence"],
                ["evidence", "answer"],
            ],
        },
        "dagig_reward_targets": weights,
        "prompt": prompt_text(row),
        "target": target,
        "target_segments": segments,
        "segment_weights": weights,
        "loss_weight": sum(weights.values()) / len(weights),
        "messages": [
            {"role": "user", "content": prompt_text(row)},
            {"role": "assistant", "content": target},
        ],
        "images": [image_path],
        "full_image_path": image_path,
        "crop_image_path": "",
        "evidences": row.get("evidences", []),
        "visual_anchor": teacher(row).get("visual_anchor") or row.get("semantic_anchor"),
        "query_specificity": teacher(row).get("query_specificity"),
        "evidence_support": teacher(row).get("evidence_support"),
        "source_grounding_result": {
            "best_score": grounding_result.get("best_score"),
            "iou": grounding_result.get("iou"),
            "center_hit": grounding_result.get("center_hit"),
            "detected": grounding_result.get("detected"),
        },
    }


def load_grounding_map(path: str) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in load_jsonl(path)}


def build_split(package_root: Path, split: str, grounding_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = load_jsonl(package_root / HARD_FILES[split])
    out = []
    for row in rows:
        sample_id = str(row.get("sample_id"))
        if sample_id not in grounding_map:
            raise KeyError(f"missing GroundingDINO result for {sample_id}")
        out.append(make_example(package_root, row, grounding_map[sample_id]))
    return out


def main() -> int:
    args = parse_args()
    package_root = Path(args.package_root).resolve()
    out_dir = Path(args.out_dir)
    split_grounding = {
        "train": load_grounding_map(args.grounding_train),
        "dev": load_grounding_map(args.grounding_dev),
        "test": load_grounding_map(args.grounding_test),
    }
    splits = {split: build_split(package_root, split, split_grounding[split]) for split in ("train", "dev", "test")}
    all_rows = [*splits["train"], *splits["dev"], *splits["test"]]
    write_jsonl(out_dir / "ground_action_train.jsonl", splits["train"])
    write_jsonl(out_dir / "ground_action_dev.jsonl", splits["dev"])
    write_jsonl(out_dir / "ground_action_test.jsonl", splits["test"])
    write_jsonl(out_dir / "ground_action_train_AB_clean_split.jsonl", all_rows)
    summary = {split: len(rows) for split, rows in splits.items()} | {"all": len(all_rows)}
    (out_dir / "ground_action_build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

