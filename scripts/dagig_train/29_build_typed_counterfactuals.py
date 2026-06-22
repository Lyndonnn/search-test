#!/usr/bin/env python3
"""Build typed counterfactuals for grounded DAG-IG edge scoring."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from grounded_pipeline_utils import (
    bbox_iou,
    get_gold_bbox,
    image_size,
    load_jsonl,
    normalize_bbox,
    row_image_path,
    tokenize,
    write_json,
    write_jsonl,
)


SPLIT_FILES = {
    "train": "data/dagig_rn03_10_grounded_rl/grounded_rl_train.jsonl",
    "dev": "data/dagig_rn03_10_grounded_rl/grounded_rl_dev.jsonl",
    "test": "data/dagig_rn03_10_grounded_rl/grounded_rl_test.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/dagig_rn03_10_grounded_rl")
    parser.add_argument("--out_dir", default="data/dagig_rn03_10_counterfactuals")
    parser.add_argument("--corpus_jsonl", default="data/dagig_rn03_10_retrieval/corpus.jsonl")
    parser.add_argument("--targets_json", default="data/dagig_rn03_10_retrieval/targets.json")
    parser.add_argument("--package_root", default="data/dagig_rn03_10_ground_expr_v3_full")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    rows = []
    for row in load_jsonl(path):
        text = str(row.get("text") or row.get("contents") or row.get("content") or "").strip()
        if text:
            rows.append(row | {"text": text, "doc_id": str(row.get("doc_id") or row.get("id") or len(rows))})
    return rows


def load_targets(path: str | Path) -> dict[str, set[str]]:
    if not Path(path).is_file():
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {str(k): {str(x) for x in v} for k, v in payload.items() if isinstance(v, list)}


def normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def remove_anchor(text: str, *anchors: str) -> str:
    out = str(text or "")
    for anchor in anchors:
        for tok in sorted(set(tokenize(anchor)), key=len, reverse=True):
            if len(tok) < 2:
                continue
            out = re.sub(rf"\b{re.escape(tok)}\b", "", out, flags=re.I)
        if anchor:
            out = re.sub(re.escape(anchor), "", out, flags=re.I)
    out = re.sub(r"\s+", " ", out).strip(" ,;:-")
    return out or "the visual target in the image"


def mask_observation(text: str, *anchors: str) -> str:
    out = str(text or "")
    for anchor in anchors:
        if anchor:
            out = re.sub(re.escape(anchor), "[MASK]", out, flags=re.I)
        for tok in sorted(set(tokenize(anchor)), key=len, reverse=True):
            if len(tok) >= 2:
                out = re.sub(rf"\b{re.escape(tok)}\b", "[MASK]", out, flags=re.I)
    out = re.sub(r"\b[A-Z][A-Za-z0-9&.-]{2,}\b", "[MASK]", out)
    out = re.sub(r"\b\d[\d,./:-]*\b", "[MASK]", out)
    return re.sub(r"\s+", " ", out).strip() or "The grounded region shows [MASK]."


def clamp_box(box: list[float], width: int, height: int) -> list[float] | None:
    return normalize_bbox(box, width, height)


def shifted_box(gold: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = gold
    bw = x2 - x1
    bh = y2 - y1
    candidates = [
        [x1 + 1.35 * bw, y1, x2 + 1.35 * bw, y2],
        [x1 - 1.35 * bw, y1, x2 - 1.35 * bw, y2],
        [x1, y1 + 1.35 * bh, x2, y2 + 1.35 * bh],
        [x1, y1 - 1.35 * bh, x2, y2 - 1.35 * bh],
    ]
    valid = [box for box in (clamp_box(c, width, height) for c in candidates) if box and bbox_iou(box, gold) < 0.05]
    if valid:
        return valid[0]
    return area_matched_box(gold, width, height, random.Random(0))


def area_matched_box(gold: list[float], width: int, height: int, rng: random.Random) -> list[float]:
    bw = max(4.0, gold[2] - gold[0])
    bh = max(4.0, gold[3] - gold[1])
    for _ in range(100):
        x1 = rng.uniform(0, max(1.0, width - bw))
        y1 = rng.uniform(0, max(1.0, height - bh))
        box = [x1, y1, x1 + bw, y1 + bh]
        if bbox_iou(box, gold) < 0.02:
            return box
    x1 = 0.0 if gold[0] > width / 2 else max(0.0, width - bw)
    y1 = 0.0 if gold[1] > height / 2 else max(0.0, height - bh)
    return [x1, y1, min(width, x1 + bw), min(height, y1 + bh)]


def nearest_len_doc(corpus: list[dict[str, Any]], target_len: int, exclude_sample: str, rng: random.Random) -> dict[str, Any]:
    candidates = [doc for doc in corpus if str(doc.get("sample_id")) != exclude_sample]
    if not candidates:
        return {"doc_id": "", "text": ""}
    sample = rng.sample(candidates, min(len(candidates), 100))
    return min(sample, key=lambda doc: abs(len(str(doc.get("text", ""))) - target_len))


def query_overlap(query: str, text: str) -> int:
    return len(set(tokenize(query)) & set(tokenize(text)))


def similar_wrong_doc(corpus: list[dict[str, Any]], query: str, sample_id: str, target_ids: set[str]) -> dict[str, Any]:
    candidates = [
        doc
        for doc in corpus
        if str(doc.get("sample_id")) != sample_id and str(doc.get("doc_id")) not in target_ids
    ]
    if not candidates:
        return {"doc_id": "", "text": ""}
    return max(candidates, key=lambda doc: query_overlap(query, str(doc.get("text", ""))))


def nonsupport_doc(corpus: list[dict[str, Any]], sample_id: str, target_ids: set[str]) -> dict[str, Any]:
    for doc in corpus:
        if str(doc.get("sample_id")) == sample_id and str(doc.get("doc_id")) not in target_ids and not doc.get("answer_supported"):
            return doc
    for doc in corpus:
        if str(doc.get("doc_id")) not in target_ids:
            return doc
    return {"doc_id": "", "text": ""}


def hard_negative_anchor(row: dict[str, Any], all_rows: list[dict[str, Any]], rng: random.Random) -> str:
    candidates = [
        str(other.get("semantic_anchor") or other.get("visual_anchor") or "").strip()
        for other in all_rows
        if other.get("sample_id") != row.get("sample_id") and str(other.get("semantic_anchor") or other.get("visual_anchor") or "").strip()
    ]
    return rng.choice(candidates) if candidates else "a different organization"


def build_row(
    row: dict[str, Any],
    all_rows: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    targets: dict[str, set[str]],
    package_root: str,
    rng: random.Random,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or "")
    anchor = str(row.get("semantic_anchor") or "").strip()
    visual_anchor = str(row.get("visual_anchor") or "").strip()
    ground = str(row.get("teacher_ground_expression") or row.get("ground_expression") or "").strip()
    observe = str(row.get("teacher_observation") or "").strip()
    search = str(row.get("teacher_search_query") or "").strip()
    evidence = str(row.get("teacher_evidence_text") or row.get("evidence") or "").strip()
    image_path = row_image_path(package_root, row)
    width, height = image_size(image_path)
    gold = normalize_bbox(get_gold_bbox(row), width, height)
    if gold is None:
        raise ValueError(f"Invalid gold bbox for sample_id={sample_id}")
    wrong_anchor = hard_negative_anchor(row, all_rows, rng)
    shuffled_obs = str(rng.choice([r for r in all_rows if r.get("sample_id") != sample_id]).get("teacher_observation") or "")
    target_ids = targets.get(sample_id, set())
    nonsupport = nonsupport_doc(corpus, sample_id, target_ids)
    similar = similar_wrong_doc(corpus, search, sample_id, target_ids)
    random_doc = nearest_len_doc(corpus, len(evidence), sample_id, rng)
    generic_query_terms = " ".join(tok for tok in tokenize(search) if tok not in set(tokenize(anchor)) and len(tok) >= 5)[:120]

    return {
        "sample_id": sample_id,
        "split": row.get("split"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "semantic_anchor": anchor,
        "visual_anchor": visual_anchor,
        "clean_rn_image_path": row.get("clean_rn_image_path"),
        "clean_rn_image_relpath_from_package": row.get("clean_rn_image_relpath_from_package"),
        "gold_bbox_pixel_xyxy": gold,
        "image_width": width,
        "image_height": height,
        "counterfactuals": {
            "ground_cf": [
                {"type": "generic_expression", "text": "the main object or region in the image"},
                {"type": "entity_removed_expression", "text": remove_anchor(ground, anchor, visual_anchor)},
                {"type": "same_image_wrong_object_expression", "text": f"a different visible object away from the {visual_anchor or 'target'}"},
            ],
            "crop_cf": [
                {"type": "nearby_wrong_box", "bbox_pixel_xyxy": shifted_box(gold, width, height)},
                {"type": "area_matched_wrong_box", "bbox_pixel_xyxy": area_matched_box(gold, width, height, rng)},
                {"type": "full_image", "bbox_pixel_xyxy": [0.0, 0.0, float(width), float(height)]},
            ],
            "observe_cf": [
                {"type": "masked_observation", "text": mask_observation(observe, anchor, visual_anchor)},
                {"type": "shuffled_same_type_observation", "text": shuffled_obs or "The grounded region shows a different visual clue."},
                {"type": "generic_observation", "text": "The grounded region shows a visible object, but its identity is unclear."},
            ],
            "search_cf": [
                {"type": "entity_removed_query", "text": remove_anchor(search, anchor, visual_anchor)},
                {"type": "generic_query", "text": generic_query_terms or "official website information answer"},
                {"type": "hard_negative_entity_query", "text": f"{wrong_anchor} {remove_anchor(search, anchor, visual_anchor)}".strip()},
            ],
            "evidence_cf": [
                {
                    "type": "non_supporting_doc",
                    "doc_id": str(nonsupport.get("doc_id", "")),
                    "text": str(nonsupport.get("text", "")),
                },
                {
                    "type": "semantically_similar_wrong_doc",
                    "doc_id": str(similar.get("doc_id", "")),
                    "text": str(similar.get("text", "")),
                },
                {
                    "type": "random_same_length_evidence_bundle",
                    "doc_id": str(random_doc.get("doc_id", "")),
                    "text": str(random_doc.get("text", "")),
                },
            ],
        },
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(args.corpus_jsonl)
    targets = load_targets(args.targets_json)
    rng = random.Random(args.seed)
    summaries = []
    for split in ["train", "dev", "test"]:
        path = Path(args.data_dir) / f"grounded_rl_{split}.jsonl"
        rows = load_jsonl(path, limit=args.limit)
        split_rng = random.Random(args.seed + len(split))
        out_rows = [build_row(row, rows, corpus, targets, args.package_root, split_rng) for row in rows]
        out_path = out_dir / f"counterfactual_{split}.jsonl"
        write_jsonl(out_path, out_rows)
        summaries.append({"split": split, "rows": len(out_rows), "output": str(out_path)})
    write_json(out_dir / "counterfactual_summary.json", {"splits": summaries, "seed": args.seed})
    print(json.dumps({"splits": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
