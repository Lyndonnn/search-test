#!/usr/bin/env python3
"""Inspect a DAG-IG Pix2Fact training package before any training.

The script intentionally fails loudly on missing images. This prevents silently
turning the visual grounding ablation into text-only training.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "MANIFEST.json",
    "IMAGE_MANIFEST.json",
    "dagig_relabel/qwen_dagig_reward_labeled_30_with_image_paths.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package_dir", required=True, help="Unzipped package directory containing MANIFEST.json.")
    parser.add_argument("--output_json", default="", help="Optional path for machine-readable inspection report.")
    parser.add_argument("--print_examples", type=int, default=3)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_no} in {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Expected object at {path}:{line_no}, got {type(row)}")
            rows.append(row)
    return rows


def resolve_package_path(package_dir: Path, relative_path: str) -> Path:
    path = package_dir / relative_path
    return path.resolve()


def inspect(package_dir: Path) -> dict[str, Any]:
    if not package_dir.is_dir():
        raise FileNotFoundError(f"package_dir does not exist or is not a directory: {package_dir}")

    missing_required = [name for name in REQUIRED_FILES if not (package_dir / name).is_file()]
    if missing_required:
        raise FileNotFoundError(f"Missing required package files: {missing_required}")

    manifest = load_json(package_dir / "MANIFEST.json")
    image_manifest = load_json(package_dir / "IMAGE_MANIFEST.json")
    main_rel = manifest.get("main_training_file") or "dagig_relabel/qwen_dagig_reward_labeled_30_with_image_paths.jsonl"
    main_path = package_dir / str(main_rel)
    if not main_path.is_file():
        raise FileNotFoundError(f"Main training file missing: {main_path}")

    rows = load_jsonl(main_path)
    source_counts = Counter(str(row.get("final_seed_source", "")) for row in rows)
    audit_counts = Counter(str(row.get("audit_severity", "")) for row in rows)

    missing_images: list[dict[str, str]] = []
    image_kinds = ["qwen_crop_readable", "qwen_full_resized", "crop_fixed", "full_original"]
    for row in rows:
        image_paths = row.get("package_image_paths") or {}
        if not isinstance(image_paths, dict):
            missing_images.append({"sample_id": str(row.get("sample_id")), "kind": "package_image_paths", "path": "<not-dict>"})
            continue
        for kind in image_kinds:
            rel = image_paths.get(kind)
            if not rel:
                missing_images.append({"sample_id": str(row.get("sample_id")), "kind": kind, "path": "<missing-field>"})
                continue
            abs_path = resolve_package_path(package_dir, str(rel))
            if not abs_path.is_file():
                missing_images.append({"sample_id": str(row.get("sample_id")), "kind": kind, "path": str(abs_path)})

    report = {
        "package_dir": str(package_dir.resolve()),
        "manifest_package_name": manifest.get("package_name"),
        "manifest_n_clean_seed": manifest.get("n_clean_seed"),
        "main_training_file": str(main_path),
        "n_rows": len(rows),
        "source_counts": dict(source_counts),
        "audit_counts": dict(audit_counts),
        "image_manifest_keys": sorted(image_manifest.keys()) if isinstance(image_manifest, dict) else [],
        "missing_image_count": len(missing_images),
        "missing_images": missing_images[:20],
        "required_files_ok": True,
    }
    if missing_images:
        raise FileNotFoundError(json.dumps(report, ensure_ascii=False, indent=2))
    return report | {"examples": rows[:3]}


def main() -> None:
    args = parse_args()
    package_dir = Path(args.package_dir).expanduser().resolve()
    report = inspect(package_dir)
    printable = dict(report)
    examples = printable.pop("examples", [])
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    for i, row in enumerate(examples[: args.print_examples]):
        print(f"\nEXAMPLE {i}")
        print(
            json.dumps(
                {
                    "sample_id": row.get("sample_id"),
                    "final_seed_source": row.get("final_seed_source"),
                    "audit_severity": row.get("audit_severity"),
                    "question": row.get("question"),
                    "qwen_local_observation": row.get("qwen_local_observation"),
                    "qwen_search_query": row.get("qwen_search_query"),
                    "answer_target": row.get("answer_target"),
                    "package_image_paths": row.get("package_image_paths"),
                    "bounded_components": row.get("bounded_components"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
