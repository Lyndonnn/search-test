#!/usr/bin/env python3
"""Validate ground-action SFT data before training."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from grounded_pipeline_utils import (
    FORBIDDEN_IMAGE_TERMS,
    PACKAGE_DIR,
    REVIEW_FILES,
    RESULT_ROOT,
    extract_tag,
    has_forbidden_image_term,
    image_size,
    load_jsonl,
    md_table,
    write_json,
)


EXPECTED = {"train": 458, "dev": 98, "test": 64}
TAGS = ["ground", "observe", "search_decision", "search", "evidence", "answer"]
FORBIDDEN_TARGET_PATTERNS = [
    r"bbox",
    r"bbox_pixel_xyxy",
    r"<tool_result>",
    r"red\s*box",
    r"red-box",
    r"annotation",
    r"annotated",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/dagig_rn03_10_grounded")
    parser.add_argument("--package_root", default=str(PACKAGE_DIR))
    parser.add_argument("--out_dir", default=str(RESULT_ROOT))
    return parser.parse_args()


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def answer_in_ground(answer: Any, ground: str) -> bool:
    answer_norm = norm(str(answer or ""))
    ground_norm = norm(ground)
    return bool(answer_norm and len(answer_norm) >= 2 and answer_norm in ground_norm)


def validate_row(row: dict[str, Any], failures: list[str]) -> None:
    sample_id = str(row.get("sample_id"))
    image = str(row.get("clean_rn_image_path") or "")
    if not image:
        failures.append(f"{sample_id}: missing clean_rn_image_path")
    else:
        path = Path(image)
        if not path.is_file():
            failures.append(f"{sample_id}: image missing {path}")
        else:
            try:
                image_size(path)
            except Exception as exc:
                failures.append(f"{sample_id}: image cannot open: {exc}")
        rel = str(row.get("clean_rn_image_relpath_from_package") or image)
        if has_forbidden_image_term(rel):
            failures.append(f"{sample_id}: image path contains forbidden term {FORBIDDEN_IMAGE_TERMS}: {rel}")

    target = str(row.get("model_target_text") or row.get("target") or "")
    for tag in TAGS:
        if not extract_tag(target, tag):
            failures.append(f"{sample_id}: missing target tag <{tag}>")
    lower = target.lower()
    for pattern in FORBIDDEN_TARGET_PATTERNS:
        if re.search(pattern, lower):
            failures.append(f"{sample_id}: model_target_text contains forbidden pattern {pattern}")
    if answer_in_ground(row.get("answer"), extract_tag(target, "ground")):
        failures.append(f"{sample_id}: <ground> leaks final answer")
    tool = row.get("tool_result_grounding")
    if not isinstance(tool, dict) or tool.get("role") != "environment_observation_not_model_target":
        failures.append(f"{sample_id}: tool_result_grounding.role invalid")


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    package_root = Path(args.package_root)
    out_dir = Path(args.out_dir)
    failures: list[str] = []
    splits: dict[str, list[dict[str, Any]]] = {}
    for split, expected in EXPECTED.items():
        path = data_dir / f"ground_action_{split}.jsonl"
        if not path.is_file():
            failures.append(f"missing {path}")
            splits[split] = []
            continue
        rows = load_jsonl(path)
        splits[split] = rows
        if len(rows) != expected:
            failures.append(f"{split} count {len(rows)} != {expected}")
        for row in rows:
            validate_row(row, failures)

    review_path = package_root / REVIEW_FILES["all"]
    review_ids = {str(row.get("sample_id")) for row in load_jsonl(review_path)} if review_path.is_file() else set()
    train_ids = {str(row.get("sample_id")) for row in splits.get("train", [])}
    overlap = train_ids & review_ids
    if overlap:
        failures.append(f"review-needed sample_id present in training data: {len(overlap)}")

    rows = [
        {"check": "train/dev/test counts", "expected": "458/98/64", "actual": f"{len(splits.get('train', []))}/{len(splits.get('dev', []))}/{len(splits.get('test', []))}", "ok": all(len(splits.get(k, [])) == v for k, v in EXPECTED.items())},
        {"check": "review-needed excluded from train", "expected": 0, "actual": len(overlap), "ok": len(overlap) == 0},
        {"check": "row-level failures", "expected": 0, "actual": len(failures), "ok": len(failures) == 0},
    ]
    payload = {
        "ok": not failures,
        "counts": {split: len(rows_) for split, rows_ in splits.items()},
        "review_train_overlap": len(overlap),
        "failures": failures,
    }
    write_json(out_dir / "ground_action_data_validation.json", payload)
    md = "# Ground-Action SFT Data Validation\n\n" + md_table(rows, ["check", "expected", "actual", "ok"]) + "\n"
    if failures:
        md += "\n## Failures\n\n" + "\n".join(f"- {item}" for item in failures[:200]) + "\n"
    (out_dir / "ground_action_data_validation.md").write_text(md, encoding="utf-8")
    print(f"wrote {out_dir / 'ground_action_data_validation.json'}")
    print(f"wrote {out_dir / 'ground_action_data_validation.md'}")
    if failures:
        print(f"FAILED checks: {len(failures)}")
        return 2
    print("ground-action SFT data validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
