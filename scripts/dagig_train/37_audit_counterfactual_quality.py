#!/usr/bin/env python3
"""Audit typed counterfactual quality for DAG-IG v2."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from grounded_pipeline_utils import bbox_area, center_hit, load_jsonl, md_table, tokenize, write_csv


SPLITS = ["train", "dev", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactual_dir", default="data/dagig_rn03_10_counterfactuals")
    parser.add_argument("--out_dir", default="results/dagig_rn03_10_counterfactual_dagig_v2")
    return parser.parse_args()


def norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def answer_leak(answer: Any, text: Any) -> bool:
    ans = norm(answer)
    body = norm(text)
    compact = ans.replace(" ", "")
    min_len = 2 if compact.isdigit() else 3
    return bool(ans and len(compact) >= min_len and ans in body)


def duplicate_texts(items: list[dict[str, Any]]) -> bool:
    texts = [norm(item.get("text")) for item in items if str(item.get("text") or "").strip()]
    return len(texts) != len(set(texts))


def box_contains_gold_center(box: list[float] | None, gold: list[float] | None) -> bool:
    return center_hit(box, gold)


def area_ratio(box: list[float] | None, gold: list[float] | None) -> float | None:
    if not box or not gold:
        return None
    g = bbox_area(gold)
    if g <= 0:
        return None
    return bbox_area(box) / g


def audit_row(row: dict[str, Any]) -> dict[str, Any]:
    cfs = row.get("counterfactuals") if isinstance(row.get("counterfactuals"), dict) else {}
    answer = row.get("answer")
    gold = row.get("gold_bbox_pixel_xyxy")
    reasons: list[str] = []
    warnings: list[str] = []

    ground_cf = cfs.get("ground_cf") if isinstance(cfs.get("ground_cf"), list) else []
    observe_cf = cfs.get("observe_cf") if isinstance(cfs.get("observe_cf"), list) else []
    search_cf = cfs.get("search_cf") if isinstance(cfs.get("search_cf"), list) else []
    evidence_cf = cfs.get("evidence_cf") if isinstance(cfs.get("evidence_cf"), list) else []
    crop_cf = cfs.get("crop_cf") if isinstance(cfs.get("crop_cf"), list) else []

    if len(ground_cf) < 3:
        reasons.append("missing_ground_cf")
    if duplicate_texts(ground_cf):
        reasons.append("duplicate_ground_cf")
    if any(answer_leak(answer, item.get("text")) for item in ground_cf):
        reasons.append("answer_leak_ground_cf")
    if any(len(tokenize(str(item.get("text") or ""))) < 2 for item in ground_cf):
        warnings.append("short_ground_cf")

    wrong_crops = [item.get("bbox_pixel_xyxy") for item in crop_cf if isinstance(item.get("bbox_pixel_xyxy"), list)]
    if len(wrong_crops) < 2:
        reasons.append("missing_crop_cf")
    non_full_wrong = [box for box in wrong_crops if area_ratio(box, gold) is not None and (area_ratio(box, gold) or 0.0) < 10.0]
    if non_full_wrong and all(box_contains_gold_center(box, gold) for box in non_full_wrong):
        reasons.append("crop_cf_contains_gold_center")
    ratios = [area_ratio(box, gold) for box in non_full_wrong]
    if ratios and not any(0.5 <= float(r) <= 2.0 for r in ratios if r is not None):
        warnings.append("no_area_matched_crop")

    if len(observe_cf) < 3:
        reasons.append("missing_observe_cf")
    if duplicate_texts(observe_cf):
        reasons.append("duplicate_observe_cf")
    if any(answer_leak(answer, item.get("text")) for item in observe_cf):
        reasons.append("answer_leak_observe_cf")

    if len(search_cf) < 3:
        reasons.append("missing_search_cf")
    if duplicate_texts(search_cf):
        reasons.append("duplicate_search_cf")
    if any(answer_leak(answer, item.get("text")) for item in search_cf):
        reasons.append("answer_leak_search_cf")
    generic = next((str(item.get("text") or "") for item in search_cf if item.get("type") == "generic_query"), "")
    if len(tokenize(generic)) > 8:
        warnings.append("generic_query_too_specific")

    if len(evidence_cf) < 3:
        reasons.append("missing_evidence_cf")
    texts = [str(item.get("text") or "").strip() for item in evidence_cf]
    if any(not text for text in texts):
        reasons.append("empty_evidence_cf")
    if texts and all(answer_leak(answer, text) for text in texts):
        reasons.append("all_evidence_cf_answer_leak")
    if duplicate_texts(evidence_cf):
        warnings.append("duplicate_evidence_cf")

    quality = "pass" if not reasons else "fail"
    return {
        "sample_id": row.get("sample_id"),
        "split": row.get("split"),
        "overall_cf_quality": quality,
        "fail_reasons": ";".join(reasons),
        "warnings": ";".join(warnings),
        "n_ground_cf": len(ground_cf),
        "n_crop_cf": len(crop_cf),
        "n_observe_cf": len(observe_cf),
        "n_search_cf": len(search_cf),
        "n_evidence_cf": len(evidence_cf),
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    all_failed: list[str] = []
    for split in SPLITS:
        rows = load_jsonl(Path(args.counterfactual_dir) / f"counterfactual_{split}.jsonl")
        audits = [audit_row(row) for row in rows]
        write_csv(out_dir / f"counterfactual_quality_audit_{split}.csv", audits)
        fail = [row for row in audits if row["overall_cf_quality"] == "fail"]
        all_failed.extend(str(row["sample_id"]) for row in fail)
        summary_rows.append(
            {
                "split": split,
                "n": len(audits),
                "pass": len(audits) - len(fail),
                "fail": len(fail),
                "fail_rate": len(fail) / max(1, len(audits)),
            }
        )
    (out_dir / "counterfactual_quality_failed_sample_ids.json").write_text(
        json.dumps(sorted(set(all_failed)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = [
        "# Counterfactual Quality Audit",
        "",
        "Rows marked `overall_cf_quality=fail` should be excluded from DAG-IG v2 scoring/training.",
        "",
        md_table(summary_rows),
        "",
        f"Failed sample ids: {len(set(all_failed))}",
    ]
    (out_dir / "counterfactual_quality_audit_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
