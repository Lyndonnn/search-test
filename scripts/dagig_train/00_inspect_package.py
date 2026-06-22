#!/usr/bin/env python3
"""Inspect and sanity-check the clean Pix2Fact-DAGIG package.

This script is intentionally strict: missing model-input images, Tier C/D rows
in the main train file, or missing action rewards fail before any training job
can start.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


DEFAULT_PACKAGE_NAME = "pix2fact_dagig_1k_gpt54_teacher_clean_package"
DEFAULT_MAIN_FILE = "data/pix2fact_dagig_train_AB_clean_split.jsonl"
REQUIRED_TOP_LEVEL = ["MANIFEST.json", "IMAGE_MANIFEST.json", "README.md"]
REQUIRED_ACTION_REWARDS = ["observe_crop", "search_query", "evidence_selection", "answer"]
IMAGE_KEYS = ["full_model_input", "crop_model_input"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package_zip", default="", help="Optional zip to unpack if package_dir is absent.")
    parser.add_argument(
        "--package_dir",
        default=f"data/{DEFAULT_PACKAGE_NAME}",
        help="Unzipped package directory containing MANIFEST.json.",
    )
    parser.add_argument("--main_file", default="", help="Path relative to package_dir; defaults to MANIFEST main file.")
    parser.add_argument("--out_dir", default="results/dagig_train/sanity")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--contact_sheet_n", type=int, default=12)
    parser.add_argument("--print_examples", type=int, default=3)
    parser.add_argument("--image_keys", nargs="+", default=IMAGE_KEYS)
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
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected object at {path}:{line_no}, got {type(row)}")
            rows.append(row)
    return rows


def maybe_unpack(package_zip: str, package_dir: Path) -> None:
    if package_dir.is_dir():
        return
    if not package_zip:
        raise FileNotFoundError(f"package_dir does not exist and --package_zip was not provided: {package_dir}")
    zip_path = Path(package_zip).expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"package_zip does not exist: {zip_path}")
    package_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(package_dir)


def teacher(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("gpt54_teacher")
    if not isinstance(value, dict):
        raise ValueError(f"Missing gpt54_teacher for sample_id={row.get('sample_id')}")
    return value


def tier(row: dict[str, Any]) -> str:
    return str(teacher(row).get("tier", "")).strip()


def training_weight(row: dict[str, Any]) -> float:
    value = teacher(row).get("training_weight")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid training_weight={value!r} for sample_id={row.get('sample_id')}") from exc


def resolve_image_path(package_dir: Path, row: dict[str, Any], key: str) -> Path:
    package_paths = row.get("package_image_paths")
    if isinstance(package_paths, dict) and package_paths.get(key):
        path = Path(str(package_paths[key]))
        return path if path.is_absolute() else package_dir / path

    image_paths = row.get("image_paths")
    if isinstance(image_paths, dict) and image_paths.get(key):
        raw = Path(str(image_paths[key]))
        if raw.is_file():
            return raw
        remapped = package_dir / "images" / key / raw.name
        if remapped.is_file():
            return remapped
        return remapped

    raise FileNotFoundError(f"Missing image path key={key} for sample_id={row.get('sample_id')}")


def verify_image(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Image missing: {path}")
    with Image.open(path) as img:
        img.verify()
    with Image.open(path) as img:
        return img.size


def validate_rows(package_dir: Path, rows: list[dict[str, Any]], image_keys: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tier_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    weight_counts: Counter[str] = Counter()
    reward_missing: list[dict[str, str]] = []
    image_sizes: list[dict[str, Any]] = []

    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        row_tier = tier(row)
        tier_counts[row_tier] += 1
        split_counts[str(row.get("split", ""))] += 1
        weight_counts[str(training_weight(row))] += 1
        if row_tier not in {"A", "B"}:
            raise ValueError(f"Main training file contains non-A/B tier row: sample_id={sample_id} tier={row_tier}")
        if not bool(teacher(row).get("keep_for_training", False)):
            raise ValueError(f"Main training file contains keep_for_training=false row: sample_id={sample_id}")

        rewards = teacher(row).get("action_rewards")
        if not isinstance(rewards, dict):
            reward_missing.append({"sample_id": sample_id, "field": "action_rewards"})
        else:
            for key in REQUIRED_ACTION_REWARDS:
                if key not in rewards:
                    reward_missing.append({"sample_id": sample_id, "field": f"action_rewards.{key}"})

        for image_key in image_keys:
            path = resolve_image_path(package_dir, row, image_key)
            width, height = verify_image(path)
            image_sizes.append({"sample_id": sample_id, "key": image_key, "path": str(path), "width": width, "height": height})

    if reward_missing:
        raise ValueError(f"Missing required action rewards: {reward_missing[:20]}")

    summary = {
        "n_rows": len(rows),
        "tier_counts": dict(tier_counts),
        "split_counts": dict(split_counts),
        "training_weight_counts": dict(weight_counts),
        "image_rows_verified": len(image_sizes),
        "image_size_preview": image_sizes[:10],
    }
    return summary, image_sizes


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    safe = text.encode("ascii", "ignore").decode("ascii")
    draw.text(xy, safe[:80], fill=(10, 10, 10))


def make_contact_sheet(package_dir: Path, rows: list[dict[str, Any]], out_path: Path, n: int, seed: int, image_keys: list[str]) -> None:
    rng = random.Random(seed)
    sample = rng.sample(rows, k=min(n, len(rows)))
    cell_w, cell_h = 520, 360
    cols = 3
    rows_n = (len(sample) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows_n * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, row in enumerate(sample):
        x0 = (idx % cols) * cell_w
        y0 = (idx // cols) * cell_h
        full = Image.open(resolve_image_path(package_dir, row, image_keys[0])).convert("RGB")
        full.thumbnail((250, 220))
        sheet.paste(full, (x0 + 10, y0 + 10))
        if len(image_keys) > 1:
            crop = Image.open(resolve_image_path(package_dir, row, image_keys[1])).convert("RGB")
            crop.thumbnail((220, 220))
            sheet.paste(crop, (x0 + 280, y0 + 10))
        t = teacher(row)
        draw_text(draw, (x0 + 10, y0 + 240), f"{row.get('sample_id')} split={row.get('split')} tier={t.get('tier')}")
        draw_text(draw, (x0 + 10, y0 + 262), f"Q: {row.get('question', '')}")
        draw_text(draw, (x0 + 10, y0 + 284), f"Anchor: {t.get('visual_anchor', '')}")
        draw_text(draw, (x0 + 10, y0 + 306), f"Query: {t.get('repaired_search_query', '')}")
        draw_text(draw, (x0 + 10, y0 + 328), f"Answer: {row.get('answer', '')}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def write_random_examples(package_dir: Path, rows: list[dict[str, Any]], out_path: Path, n: int, seed: int, image_keys: list[str]) -> None:
    rng = random.Random(seed)
    sample = rng.sample(rows, k=min(n, len(rows)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Pix2Fact-DAGIG Random Examples\n\n")
        for row in sample:
            t = teacher(row)
            f.write(f"## {row.get('sample_id')}\n\n")
            f.write(f"- split: `{row.get('split')}`\n")
            f.write(f"- tier: `{t.get('tier')}`\n")
            f.write(f"- question: {row.get('question')}\n")
            f.write(f"- answer: `{row.get('answer')}`\n")
            f.write(f"- visual_anchor: `{t.get('visual_anchor')}`\n")
            f.write(f"- repaired_search_query: `{t.get('repaired_search_query')}`\n")
            for image_key in image_keys:
                f.write(f"- {image_key}: `{resolve_image_path(package_dir, row, image_key)}`\n")
            f.write("\n")


def first_existing_manifest(package_dir: Path) -> Path | None:
    for name in ["MANIFEST.json", "MANIFEST_PAPER_AUDITED_RN03_10.json"]:
        path = package_dir / name
        if path.is_file():
            return path
    matches = sorted(package_dir.glob("MANIFEST*.json"))
    return matches[0] if matches else None


def inspect(package_dir: Path, main_file: str, out_dir: Path, seed: int, contact_sheet_n: int, image_keys: list[str]) -> dict[str, Any]:
    if not package_dir.is_dir():
        raise FileNotFoundError(f"package_dir does not exist or is not a directory: {package_dir}")

    manifest_path = first_existing_manifest(package_dir)
    manifest = load_json(manifest_path) if manifest_path else {}
    image_manifest_path = package_dir / "IMAGE_MANIFEST.json"
    image_manifest = load_json(image_manifest_path) if image_manifest_path.is_file() else {}
    main_rel = main_file or manifest.get("main_training_file") or manifest.get("final_training_jsonl") or DEFAULT_MAIN_FILE
    main_path = Path(str(main_rel))
    if not main_path.is_absolute():
        main_path = package_dir / main_path
    if not main_path.is_file():
        raise FileNotFoundError(f"Main training file missing: {main_path}")

    rows = load_jsonl(main_path)
    summary, _image_sizes = validate_rows(package_dir, rows, image_keys)
    report = {
        "package_dir": str(package_dir.resolve()),
        "main_training_file": str(main_path.resolve()),
        "manifest_path": str(manifest_path) if manifest_path else "",
        "manifest": manifest,
        "image_manifest_top_level_keys": sorted(image_manifest.keys()) if isinstance(image_manifest, dict) else [],
        "image_keys_verified": image_keys,
        **summary,
        "required_files_ok": True,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    make_contact_sheet(package_dir, rows, out_dir / "contact_sheet.jpg", contact_sheet_n, seed, image_keys)
    write_random_examples(package_dir, rows, out_dir / "random_examples.md", contact_sheet_n, seed, image_keys)
    return report


def main() -> None:
    args = parse_args()
    package_dir = Path(args.package_dir).expanduser().resolve()
    maybe_unpack(args.package_zip, package_dir)
    report = inspect(
        package_dir,
        args.main_file,
        Path(args.out_dir).expanduser().resolve(),
        args.seed,
        args.contact_sheet_n,
        args.image_keys,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    rows = load_jsonl(Path(report["main_training_file"]))
    for i, row in enumerate(rows[: args.print_examples]):
        t = teacher(row)
        print(f"\nEXAMPLE {i}")
        print(
            json.dumps(
                {
                    "sample_id": row.get("sample_id"),
                    "split": row.get("split"),
                    "tier": t.get("tier"),
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "visual_anchor": t.get("visual_anchor"),
                    "repaired_search_query": t.get("repaired_search_query"),
                    "action_rewards": t.get("action_rewards"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
