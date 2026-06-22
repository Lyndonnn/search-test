#!/usr/bin/env python3
"""Rebuild stronger search counterfactuals and evaluate retrieval separation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from grounded_pipeline_utils import load_jsonl, md_table, tokenize, write_jsonl


SCORER_PATH = Path(__file__).with_name("26_score_grounded_rollouts.py")
SPEC = importlib.util.spec_from_file_location("grounded_rollout_scorer", SCORER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import scorer from {SCORER_PATH}")
SCORER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER_MODULE)
GroundedRewardScorer = SCORER_MODULE.GroundedRewardScorer


SPLITS = ["train", "dev", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/dagig_rn03_10_grounded_rl")
    parser.add_argument("--out_dir", default="data/dagig_rn03_10_counterfactuals_v2")
    parser.add_argument("--result_dir", default="results/dagig_rn03_10_counterfactual_dagig_v2")
    parser.add_argument("--corpus_jsonl", default="data/dagig_rn03_10_retrieval/corpus.jsonl")
    parser.add_argument("--targets_json", default="data/dagig_rn03_10_retrieval/targets.json")
    parser.add_argument("--package_root", default="data/dagig_rn03_10_ground_expr_v3_full")
    parser.add_argument("--min_real_beats_best_rate", type=float, default=0.60)
    parser.add_argument("--no_fail_on_weak", action="store_true")
    return parser.parse_args()


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def remove_anchor(text: str, *anchors: str) -> str:
    out = str(text or "")
    for anchor in anchors:
        if anchor:
            out = re.sub(re.escape(anchor), "", out, flags=re.I)
        for tok in sorted(set(tokenize(anchor)), key=len, reverse=True):
            if len(tok) >= 2:
                out = re.sub(rf"\b{re.escape(tok)}\b", "", out, flags=re.I)
    return norm(out).strip(" ,;:-") or "official website information"


def generic_query(query: str, anchor: str, visual_anchor: str) -> str:
    removed = remove_anchor(query, anchor, visual_anchor)
    toks = [tok for tok in tokenize(removed) if len(tok) >= 4 and not tok.isdigit()]
    generic = " ".join(toks[:6])
    return generic or "official website contact information"


def wrong_anchor(row: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    anchor = norm(row.get("semantic_anchor") or row.get("visual_anchor"))
    candidates = []
    for other in rows:
        if other.get("sample_id") == row.get("sample_id"):
            continue
        cand = norm(other.get("semantic_anchor") or other.get("visual_anchor"))
        if cand and cand.lower() != anchor.lower():
            candidates.append(cand)
    if not candidates:
        return "a different organization"
    target_tokens = set(tokenize(anchor))
    return max(candidates, key=lambda cand: len(target_tokens & set(tokenize(cand))))


def retrieval_score(metrics: dict[str, Any]) -> float:
    r5 = 1.0 if metrics.get("retrieval_r5") is True else 0.0
    r1 = 1.0 if metrics.get("retrieval_r1") is True else 0.0
    mrr = float(metrics.get("retrieval_mrr") or 0.0)
    return 0.45 * r5 + 0.25 * r1 + 0.30 * mrr


def query_set(row: dict[str, Any], all_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    real = norm(row.get("teacher_search_query") or (row.get("target_segments") or {}).get("search"))
    anchor = norm(row.get("semantic_anchor"))
    visual_anchor = norm(row.get("visual_anchor"))
    entity_removed = remove_anchor(real, anchor, visual_anchor)
    generic = generic_query(real, anchor, visual_anchor)
    hard = f"{wrong_anchor(row, all_rows)} {entity_removed}".strip()
    return [
        {"type": "real", "text": real},
        {"type": "entity_removed", "text": entity_removed},
        {"type": "generic", "text": generic},
        {"type": "hard_negative", "text": hard},
    ]


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    result_dir = Path(args.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    scorer = GroundedRewardScorer(
        corpus_jsonl=args.corpus_jsonl,
        targets_json=args.targets_json,
        package_root=args.package_root,
        use_dino=False,
    )
    summary = []
    all_rows_by_split = {split: load_jsonl(Path(args.data_dir) / f"grounded_rl_{split}.jsonl") for split in SPLITS}
    for split, rows in all_rows_by_split.items():
        output = []
        for row in rows:
            queries = []
            for item in query_set(row, rows):
                metrics = scorer.retrieval_metrics(row, item["text"])
                queries.append({**item, **metrics, "retrieval_score": retrieval_score(metrics)})
            real = next(item for item in queries if item["type"] == "real")
            cfs = [item for item in queries if item["type"] != "real"]
            best_cf = max(float(item["retrieval_score"]) for item in cfs) if cfs else 0.0
            mean_cf = mean(float(item["retrieval_score"]) for item in cfs) if cfs else 0.0
            output.append(
                {
                    "sample_id": row.get("sample_id"),
                    "split": split,
                    "real_query": real["text"],
                    "queries": queries,
                    "real_score": real["retrieval_score"],
                    "best_cf_score": best_cf,
                    "mean_cf_score": mean_cf,
                    "real_beats_best_cf": float(real["retrieval_score"]) > best_cf,
                    "real_beats_mean_cf": float(real["retrieval_score"]) > mean_cf,
                }
            )
        write_jsonl(out_dir / f"search_counterfactuals_v2_{split}.jsonl", output)
        rate_best = sum(1 for row in output if row["real_beats_best_cf"]) / max(1, len(output))
        rate_mean = sum(1 for row in output if row["real_beats_mean_cf"]) / max(1, len(output))
        summary.append(
            {
                "split": split,
                "n": len(output),
                "real_beats_best_cf_rate": rate_best,
                "real_beats_mean_cf_rate": rate_mean,
                "passes_best_threshold": rate_best >= args.min_real_beats_best_rate,
            }
        )
    md = [
        "# Search Counterfactual Diagnostics v2",
        "",
        md_table(summary),
        "",
        f"Minimum required real_beats_best_cf_rate: {args.min_real_beats_best_rate}",
    ]
    (result_dir / "search_counterfactual_diagnostics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.no_fail_on_weak and any(not row["passes_best_threshold"] for row in summary):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
