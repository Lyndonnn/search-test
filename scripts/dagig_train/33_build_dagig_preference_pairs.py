#!/usr/bin/env python3
"""Build DPO/rejection-SFT preference pairs from counterfactual DAG-IG scores."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from grounded_pipeline_utils import extract_segments, load_jsonl, write_jsonl


TAGS = ["ground", "observe", "search_decision", "search", "evidence", "answer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored_dir", default="results/dagig_rn03_10_counterfactual_dagig/scored")
    parser.add_argument("--data_dir", default="data/dagig_rn03_10_grounded_rl")
    parser.add_argument("--out_dir", default="data/dagig_rn03_10_counterfactual_dagig")
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--max_pairs_per_prompt", type=int, default=1)
    return parser.parse_args()


def answer_leak(row: dict[str, Any]) -> bool:
    if row.get("answer_leakage"):
        return True
    segments = row.get("prediction_segments")
    if not isinstance(segments, dict):
        segments = extract_segments(str(row.get("prediction") or ""), TAGS)
    answer = str(segments.get("answer") or "")
    leak_text = "\n".join(str(segments.get(tag, "")) for tag in ["ground", "observe", "search_decision", "search"])
    return bool(answer and len(answer) >= 2 and answer.lower() in leak_text.lower())


def eligible_high(row: dict[str, Any]) -> bool:
    return bool(
        row.get("format_valid")
        and not answer_leak(row)
        and not row.get("unsupported_answer")
        and not row.get("spurious_success")
    )


def build_pair(example: dict[str, Any], high: dict[str, Any], low: dict[str, Any]) -> dict[str, Any]:
    chosen = str(high.get("prediction") or "")
    rejected = str(low.get("prediction") or "")
    return {
        "sample_id": example.get("sample_id"),
        "split": example.get("split"),
        "prompt": example.get("prompt"),
        "images": example.get("images", []),
        "chosen": chosen,
        "rejected": rejected,
        "target": chosen,
        "target_segments": extract_segments(chosen, TAGS),
        "loss_weight": 1.0,
        "segment_weights": {
            "ground": 1.0,
            "observe": 1.0,
            "search_decision": 0.4,
            "search": 1.1,
            "evidence": 1.1,
            "answer": 0.8,
        },
        "chosen_DAGIG_total": high.get("DAGIG_total"),
        "rejected_DAGIG_total": low.get("DAGIG_total"),
        "margin": float(high.get("DAGIG_total") or 0.0) - float(low.get("DAGIG_total") or 0.0),
        "chosen_metrics": {
            key: high.get(key)
            for key in ["R_ground", "R_observe", "R_search", "R_evidence", "R_answer", "R_cost", "retrieval_r5", "evidence_support", "answer_f1"]
        },
        "rejected_metrics": {
            key: low.get(key)
            for key in ["R_ground", "R_observe", "R_search", "R_evidence", "R_answer", "R_cost", "retrieval_r5", "evidence_support", "answer_f1"]
        },
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        examples = {str(row.get("sample_id")): row for row in load_jsonl(Path(args.data_dir) / f"grounded_rl_{split}.jsonl")}
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in load_jsonl(Path(args.scored_dir) / f"{split}.jsonl"):
            groups[str(row.get("sample_id"))].append(row)
        pairs = []
        for sample_id, rows in groups.items():
            example = examples.get(sample_id)
            if example is None:
                continue
            highs = sorted([row for row in rows if eligible_high(row)], key=lambda row: float(row.get("DAGIG_total") or -999), reverse=True)
            lows = sorted(rows, key=lambda row: float(row.get("DAGIG_total") or 999))
            made = 0
            for high in highs:
                for low in lows:
                    margin = float(high.get("DAGIG_total") or 0.0) - float(low.get("DAGIG_total") or 0.0)
                    if margin >= args.margin and str(high.get("prediction") or "") != str(low.get("prediction") or ""):
                        pairs.append(build_pair(example, high, low))
                        made += 1
                        break
                if made >= args.max_pairs_per_prompt:
                    break
        out_path = out_dir / f"dagig_preference_pairs_{split}.jsonl"
        write_jsonl(out_path, pairs)
        print(json.dumps({"split": split, "pairs": len(pairs), "output": str(out_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
