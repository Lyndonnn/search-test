#!/usr/bin/env python3
"""Validate the full RN03_10 ground-expression package before training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from grounded_pipeline_utils import (
    FORBIDDEN_IMAGE_TERMS,
    HARD_FILES,
    IMAGE_SUFFIXES,
    PACKAGE_DIR,
    RESULT_ROOT,
    REVIEW_FILES,
    get_gold_bbox,
    hard_pass,
    has_forbidden_image_term,
    image_size,
    load_jsonl,
    md_table,
    row_image_path,
    write_json,
)


EXPECTED_HARD_COUNTS = {"train": 458, "dev": 98, "test": 64}
EXPECTED_HARD_TOTAL = 620
EXPECTED_REVIEW_TOTAL = 161
EXPECTED_IMAGE_COUNT = 781
REQUIRED_FIELDS = [
    "sample_id",
    "split",
    "question",
    "answer",
    "ground_expression",
    "ground_action",
    "grounding",
    "ground_tool_gold",
    "clean_rn_image_relpath_from_package",
    "ground_expression_quality",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package_root", default=str(PACKAGE_DIR))
    parser.add_argument("--out_dir", default=str(RESULT_ROOT))
    return parser.parse_args()


def fail_if(condition: bool, failures: list[str], message: str) -> None:
    if condition:
        failures.append(message)


def validate_row(package_root: Path, row: dict[str, Any], failures: list[str], check_training_path: bool) -> None:
    sample_id = row.get("sample_id", "<missing>")
    for field in REQUIRED_FIELDS:
        fail_if(field not in row or row.get(field) in (None, ""), failures, f"{sample_id}: missing {field}")
    fail_if(not hard_pass(row), failures, f"{sample_id}: ground_expression_quality.hard_pass is not true")
    try:
        get_gold_bbox(row)
    except Exception as exc:
        failures.append(f"{sample_id}: invalid ground_tool_gold bbox: {exc}")
    rel = str(row.get("clean_rn_image_relpath_from_package") or "")
    if check_training_path:
        fail_if(
            has_forbidden_image_term(rel),
            failures,
            f"{sample_id}: training clean image path contains forbidden term {FORBIDDEN_IMAGE_TERMS}: {rel}",
        )
    try:
        path = row_image_path(package_root, row)
        fail_if(not path.is_file(), failures, f"{sample_id}: missing image {path}")
        if path.is_file():
            image_size(path)
    except Exception as exc:
        failures.append(f"{sample_id}: image open failed: {exc}")


def main() -> int:
    args = parse_args()
    package_root = Path(args.package_root).resolve()
    out_dir = Path(args.out_dir)
    failures: list[str] = []
    warnings: list[str] = []

    fail_if(not package_root.is_dir(), failures, f"Missing package root: {package_root}")

    hard_rows: dict[str, list[dict[str, Any]]] = {}
    for split, rel in HARD_FILES.items():
        path = package_root / rel
        fail_if(not path.is_file(), failures, f"Missing hard-pass file: {rel}")
        hard_rows[split] = load_jsonl(path) if path.is_file() else []

    review_rows: dict[str, list[dict[str, Any]]] = {}
    for split, rel in REVIEW_FILES.items():
        path = package_root / rel
        fail_if(not path.is_file(), failures, f"Missing review-needed file: {rel}")
        review_rows[split] = load_jsonl(path) if path.is_file() else []

    split_counts = {split: len(hard_rows.get(split, [])) for split in ("train", "dev", "test")}
    for split, expected in EXPECTED_HARD_COUNTS.items():
        fail_if(split_counts.get(split) != expected, failures, f"{split} count {split_counts.get(split)} != {expected}")

    split_union = {str(row.get("sample_id")) for split in ("train", "dev", "test") for row in hard_rows.get(split, [])}
    all_hard_ids = {str(row.get("sample_id")) for row in hard_rows.get("all", [])}
    fail_if(len(split_union) != EXPECTED_HARD_TOTAL, failures, f"hard-pass unique total {len(split_union)} != {EXPECTED_HARD_TOTAL}")
    fail_if(len(all_hard_ids) != EXPECTED_HARD_TOTAL, failures, f"all_hard_pass file unique total {len(all_hard_ids)} != {EXPECTED_HARD_TOTAL}")
    fail_if(split_union != all_hard_ids, failures, "train/dev/test hard-pass union differs from all_hard_pass file")

    review_ids = {str(row.get("sample_id")) for row in review_rows.get("all", [])}
    fail_if(len(review_ids) != EXPECTED_REVIEW_TOTAL, failures, f"review-needed unique total {len(review_ids)} != {EXPECTED_REVIEW_TOTAL}")
    fail_if(bool(split_union & review_ids), failures, f"hard-pass and review-needed overlap: {len(split_union & review_ids)}")
    for row in review_rows.get("all", []):
        if hard_pass(row):
            failures.append(f"{row.get('sample_id')}: review-needed row unexpectedly has hard_pass=true")

    image_files = [p for p in (package_root / "images" / "rn_clean").rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    fail_if(len(image_files) != EXPECTED_IMAGE_COUNT, failures, f"clean RN image count {len(image_files)} != {EXPECTED_IMAGE_COUNT}")

    for split in ("train", "dev", "test"):
        for row in hard_rows.get(split, []):
            validate_row(package_root, row, failures, check_training_path=(split == "train"))

    summary = {
        "package_root": str(package_root),
        "ok": not failures,
        "hard_pass_total": len(split_union),
        "hard_pass_split_counts": split_counts,
        "review_needed_total": len(review_ids),
        "clean_rn_image_count": len(image_files),
        "hard_review_overlap": len(split_union & review_ids),
        "failures": failures,
        "warnings": warnings,
    }
    write_json(out_dir / "package_validation.json", summary)

    rows = [
        {"check": "hard-pass total", "expected": EXPECTED_HARD_TOTAL, "actual": len(split_union), "ok": len(split_union) == EXPECTED_HARD_TOTAL},
        {"check": "train/dev/test", "expected": "458/98/64", "actual": f"{split_counts.get('train')}/{split_counts.get('dev')}/{split_counts.get('test')}", "ok": split_counts == EXPECTED_HARD_COUNTS},
        {"check": "review-needed", "expected": EXPECTED_REVIEW_TOTAL, "actual": len(review_ids), "ok": len(review_ids) == EXPECTED_REVIEW_TOTAL},
        {"check": "clean RN images", "expected": EXPECTED_IMAGE_COUNT, "actual": len(image_files), "ok": len(image_files) == EXPECTED_IMAGE_COUNT},
        {"check": "hard/review disjoint", "expected": 0, "actual": len(split_union & review_ids), "ok": not (split_union & review_ids)},
    ]
    md = "# Package Validation\n\n" + md_table(rows, ["check", "expected", "actual", "ok"]) + "\n"
    if failures:
        md += "\n## Failures\n\n" + "\n".join(f"- {item}" for item in failures[:200]) + "\n"
    if warnings:
        md += "\n## Warnings\n\n" + "\n".join(f"- {item}" for item in warnings[:200]) + "\n"
    (out_dir / "package_validation.md").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "package_validation.md").write_text(md, encoding="utf-8")
    print(f"wrote {out_dir / 'package_validation.json'}")
    print(f"wrote {out_dir / 'package_validation.md'}")
    if failures:
        print(f"FAILED checks: {len(failures)}")
        return 2
    print("package validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

