#!/usr/bin/env python3
"""GroundingDINO smoke test for RN03-10 DAG-IG data.

The smoke test intentionally does one thing: load one image plus its
`ground_expression`, run GroundingDINO, and save a red-box visualization. It
does not train, relabel, or assume a fixed trajectory chain.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEXT_KEYS = [
    "ground_expression",
    "ground_expr",
    "grounding_expression",
    "grounding_expr",
    "ground_phrase",
    "phrase",
]
IMAGE_KEYS = [
    "image_path",
    "image",
    "image_file",
    "image_filename",
    "local_image_path",
    "full_image_path",
    "source_image",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="data/rn03_10/extracted")
    parser.add_argument("--sample_file", default="", help="Optional explicit JSON/JSONL/CSV metadata file.")
    parser.add_argument("--image_path", default="", help="Optional explicit image path.")
    parser.add_argument("--text_prompt", default="", help="Optional explicit grounding text prompt.")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--repo", default="third_party/GroundingDINO")
    parser.add_argument("--config", default="")
    parser.add_argument("--checkpoint", default="third_party/GroundingDINO_weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--output_dir", default="paper_artifacts/figures/groundingdino_smoke")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.20)
    return parser.parse_args()


def read_json(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ["data", "samples", "rows", "items"]:
            value = obj.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [obj]
    return []


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return read_json(path)
    if suffix in {".jsonl", ".ndjson"}:
        return read_jsonl(path)
    if suffix == ".csv":
        return read_csv(path)
    return []


def find_metadata_files(data_root: Path) -> list[Path]:
    suffixes = {".json", ".jsonl", ".ndjson", ".csv"}
    return sorted(p for p in data_root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def normalize_path(path_value: Any, data_root: Path, metadata_file: Path | None) -> Path | None:
    if not path_value:
        return None
    if isinstance(path_value, dict):
        for key in ["full_original", "qwen_full_resized", "crop_fixed", "qwen_crop_readable", "path"]:
            if key in path_value:
                return normalize_path(path_value[key], data_root, metadata_file)
        return None
    text = str(path_value)
    path = Path(text)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(data_root / path)
        if metadata_file is not None:
            candidates.append(metadata_file.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def row_text_prompt(row: dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def row_image_path(row: dict[str, Any], data_root: Path, metadata_file: Path | None) -> Path | None:
    for key in IMAGE_KEYS:
        path = normalize_path(row.get(key), data_root, metadata_file)
        if path is not None:
            return path
    package_paths = row.get("package_image_paths")
    path = normalize_path(package_paths, data_root, metadata_file)
    if path is not None:
        return path
    return None


def find_sample(data_root: Path, sample_file: str, sample_index: int) -> tuple[Path, str, dict[str, Any]]:
    metadata_files = [Path(sample_file)] if sample_file else find_metadata_files(data_root)
    checked = []
    matches: list[tuple[Path, str, dict[str, Any], Path]] = []
    for meta in metadata_files:
        if not meta.is_file():
            continue
        rows = load_rows(meta)
        checked.append((str(meta), len(rows)))
        for row in rows:
            prompt = row_text_prompt(row)
            image = row_image_path(row, data_root, meta)
            if prompt and image is not None:
                matches.append((image, prompt, row, meta))
    if matches:
        image, prompt, row, meta = matches[min(sample_index, len(matches) - 1)]
        row = dict(row)
        row["_metadata_file"] = str(meta)
        return image, prompt, row

    raise RuntimeError(
        "No metadata row with both image path and ground expression was found.\n"
        f"data_root={data_root}\n"
        f"sample_file={sample_file or '<auto>'}\n"
        f"checked_metadata_files={checked[:20]}\n"
        f"expected_text_keys={TEXT_KEYS}\n"
        f"expected_image_keys={IMAGE_KEYS} or package_image_paths\n"
        "Fallback: pass --image_path and --text_prompt explicitly."
    )


def first_image(data_root: Path) -> Path | None:
    for path in sorted(data_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            return path.resolve()
    return None


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, str, dict[str, Any]]:
    data_root = Path(args.data_root).expanduser().resolve()
    if args.image_path and args.text_prompt:
        image = Path(args.image_path).expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"Explicit image_path missing: {image}")
        return image, args.text_prompt.strip(), {"_source": "explicit_args"}
    if args.image_path or args.text_prompt:
        raise ValueError("Pass both --image_path and --text_prompt, or neither.")
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root missing: {data_root}")
    return find_sample(data_root, args.sample_file, args.sample_index)


def save_annotated(annotated_frame: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        cv2.imwrite(str(out_path), annotated_frame)
        return
    except Exception:
        pass
    try:
        from PIL import Image
        import numpy as np

        arr = np.asarray(annotated_frame)
        Image.fromarray(arr).save(out_path)
    except Exception as exc:
        raise RuntimeError(f"Could not save annotated image to {out_path}") from exc


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config = Path(args.config).expanduser().resolve() if args.config else repo / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not repo.is_dir():
        raise FileNotFoundError(f"GroundingDINO repo missing: {repo}")
    if not config.is_file():
        raise FileNotFoundError(f"GroundingDINO config missing: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"GroundingDINO checkpoint missing: {checkpoint}")

    sys.path.insert(0, str(repo))
    try:
        from groundingdino.util.inference import annotate, load_image, load_model, predict
    except Exception as exc:
        raise ImportError(
            "Could not import GroundingDINO inference utilities. "
            "Run: bash scripts/dagig_train/00_setup_groundingdino_env.sh"
        ) from exc

    image_path, text_prompt, sample = resolve_inputs(args)
    model = load_model(str(config), str(checkpoint))
    image_source, image = load_image(str(image_path))
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    annotated = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    out_path = output_dir / f"{sample.get('sample_id', 'sample')}_groundingdino_smoke.jpg"
    save_annotated(annotated, out_path)

    result = {
        "image_path": str(image_path),
        "text_prompt": text_prompt,
        "metadata_file": sample.get("_metadata_file"),
        "sample_id": sample.get("sample_id"),
        "num_boxes": int(len(boxes)),
        "scores": [float(x) for x in logits.detach().cpu().tolist()] if hasattr(logits, "detach") else [float(x) for x in logits],
        "phrases": list(phrases),
        "output_path": str(out_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
