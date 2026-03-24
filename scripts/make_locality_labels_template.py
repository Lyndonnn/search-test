#!/usr/bin/env python3
import argparse
import os
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a locality/search/action labeling template from Pix2Fact clean parquet."
    )
    parser.add_argument("--input", required=True, help="Path to Pix2Fact clean parquet.")
    parser.add_argument("--output", required=True, help="Path to output CSV.")
    return parser.parse_args()


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if hasattr(value, "item"):
        try:
            return safe_text(value.item())
        except Exception:
            pass
    return str(value).strip()


def bootstrap_evidence_type(row: pd.Series) -> str:
    hint = " ".join(
        [
            safe_text(row.get("visual_perception_type", "")),
            safe_text(row.get("category", "")),
            safe_text(row.get("image_description", "")),
        ]
    ).lower()
    if any(token in hint for token in ["logo", "brand", "emblem"]):
        return "logo"
    if any(token in hint for token in ["screen", "website", "app", "display", "monitor"]):
        return "screen"
    if any(token in hint for token in ["document", "receipt", "newspaper", "paper", "page", "map"]):
        return "document"
    if any(token in hint for token in ["text", "sign", "poster", "caption", "license plate"]):
        return "text"
    if any(token in hint for token in ["person", "face", "human", "celebrity"]):
        return "person"
    if any(token in hint for token in ["part", "detail", "fine-grained", "close-up"]):
        return "fine_grained_part"
    if any(token in hint for token in ["object", "product", "vehicle", "animal", "building"]):
        return "object"
    return "mixed"


def bootstrap_post_local_action(evidence_type: str) -> str:
    if evidence_type in {"text", "screen", "document"}:
        return "ocr"
    if evidence_type == "logo":
        return "image_search"
    return "verify"


def bootstrap_search_requirement(row: pd.Series) -> str:
    has_evidence_url = any(
        safe_text(row.get(column, ""))
        for column in ("evidence_url_1", "evidence_url_2", "evidence_url_3")
    )
    if has_evidence_url:
        return "mixed_required"
    return "search_free"


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)

    evidence_type_bootstrap = [bootstrap_evidence_type(row) for _, row in df.iterrows()]
    post_action_bootstrap = [bootstrap_post_local_action(v) for v in evidence_type_bootstrap]
    search_requirement_bootstrap = [bootstrap_search_requirement(row) for _, row in df.iterrows()]
    has_bbox = [int(bool(safe_text(row.get("bounding_box", "")))) for _, row in df.iterrows()]

    template = pd.DataFrame(
        {
            "qid": [safe_text(v) for v in df.get("qid", [])],
            "item_id": [safe_text(v) for v in df.get("item_id", [])],
            "source_dataset": [safe_text(v) for v in df.get("source_dataset", [])],
            "question": [safe_text(v) for v in df.get("question", [])],
            "answer": [safe_text(v) for v in df.get("answer", [])],
            "category": [safe_text(v) for v in df.get("category", [])],
            "visual_perception_type": [safe_text(v) for v in df.get("visual_perception_type", [])],
            "knowledge_domain": [safe_text(v) for v in df.get("knowledge_domain", [])],
            "reasoning_logic_type": [safe_text(v) for v in df.get("reasoning_logic_type", [])],
            "bounding_box": [safe_text(v) for v in df.get("bounding_box", [])],
            "evidence_1": [safe_text(v) for v in df.get("evidence_1", [])],
            "evidence_2": [safe_text(v) for v in df.get("evidence_2", [])],
            "evidence_3": [safe_text(v) for v in df.get("evidence_3", [])],
            "evidence_url_1": [safe_text(v) for v in df.get("evidence_url_1", [])],
            "evidence_url_2": [safe_text(v) for v in df.get("evidence_url_2", [])],
            "evidence_url_3": [safe_text(v) for v in df.get("evidence_url_3", [])],
            "needs_zoom_bootstrap": has_bbox,
            "needs_search_bootstrap": [int(v != "search_free") for v in search_requirement_bootstrap],
            "locality_level_bootstrap": ["local_critical" if flag else "local_helpful" for flag in has_bbox],
            "search_requirement_bootstrap": search_requirement_bootstrap,
            "evidence_type_bootstrap": evidence_type_bootstrap,
            "post_local_action_bootstrap": post_action_bootstrap,
            "locality_level": ["" for _ in range(len(df))],
            "search_requirement": ["" for _ in range(len(df))],
            "evidence_type": ["" for _ in range(len(df))],
            "post_local_action": ["" for _ in range(len(df))],
            "evidence_sufficiency": ["" for _ in range(len(df))],
            "needs_zoom": ["" for _ in range(len(df))],
            "needs_search": ["" for _ in range(len(df))],
            "human_checked": [0 for _ in range(len(df))],
            "notes": ["" for _ in range(len(df))],
        }
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    template.to_csv(args.output, index=False)
    print(f"Saved {len(template)} rows to {args.output}")


if __name__ == "__main__":
    main()
