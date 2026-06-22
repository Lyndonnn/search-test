#!/usr/bin/env python3
"""Re-score only the search edge using retrieval-aware v3 counterfactuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from grounded_pipeline_utils import load_jsonl, md_table, write_json, write_jsonl


SPLITS = ["train", "dev", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_scored_dir", default="results/dagig_rn03_10_counterfactual_dagig/scored")
    parser.add_argument("--search_cf_dir", default="data/dagig_rn03_10_counterfactuals_v3")
    parser.add_argument("--out_dir", default="results/dagig_rn03_10_counterfactual_dagig_v3")
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    return parser.parse_args()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def retrieval_score_from_metrics(metrics: dict[str, Any]) -> float:
    if metrics.get("retrieval_score") is not None:
        return clamp01(float(metrics.get("retrieval_score") or 0.0))
    r5 = 1.0 if as_bool(metrics.get("retrieval_r5", metrics.get("support@5"))) else 0.0
    r1 = 1.0 if as_bool(metrics.get("retrieval_r1", metrics.get("support@1"))) else 0.0
    mrr = float(metrics.get("retrieval_mrr", metrics.get("MRR", 0.0)) or 0.0)
    return clamp01(0.45 * r5 + 0.25 * r1 + 0.30 * mrr)


def normalize_delta(real: float, cf: float) -> float:
    return (float(real) - float(cf)) / max(1.0, abs(float(real)), abs(float(cf)))


def edge_record(real_score: float, cf_scores: list[float], gate_pass: bool) -> dict[str, Any]:
    if not cf_scores:
        cf_scores = [0.0]
    cf_best = max(float(v) for v in cf_scores)
    cf_mean = mean(float(v) for v in cf_scores)
    delta_best = normalize_delta(real_score, cf_best)
    delta_mean = normalize_delta(real_score, cf_mean)
    edge_credit = delta_best if gate_pass else min(delta_best, 0.0)
    return {
        "real_score": float(real_score),
        "cf_best_score": float(cf_best),
        "cf_mean_score": float(cf_mean),
        "delta_best": float(delta_best),
        "delta_mean": float(delta_mean),
        "edge_credit": float(edge_credit),
        "gate_pass": bool(gate_pass),
    }


def load_search_cfs(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in load_jsonl(path)}


def supported_answer(row: dict[str, Any]) -> bool:
    answer_match = float(row.get("answer_em") or 0.0) > 0.0 or float(row.get("answer_f1") or 0.0) >= 0.50
    return bool(as_bool(row.get("evidence_support")) and answer_match)


def score_split(split: str, input_scored_dir: Path, search_cf_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    scored_path = input_scored_dir / f"{split}.jsonl"
    cf_path = search_cf_dir / f"search_counterfactuals_v3_{split}.jsonl"
    if not scored_path.is_file():
        raise FileNotFoundError(scored_path)
    if not cf_path.is_file():
        raise FileNotFoundError(cf_path)
    search_cfs = load_search_cfs(cf_path)
    out_rows = []
    for row in load_jsonl(scored_path):
        sample_id = str(row.get("sample_id") or "")
        cf = search_cfs.get(sample_id, {})
        valid_items = cf.get("valid_counterfactuals") or []
        cf_scores = [retrieval_score_from_metrics(item.get("retrieval") or {}) for item in valid_items]
        pred_search = ""
        if isinstance(row.get("prediction_segments"), dict):
            pred_search = str(row["prediction_segments"].get("search") or "")
        real_score = retrieval_score_from_metrics(row)
        gate_pass = bool(
            as_bool(row.get("format_valid"))
            and not as_bool(row.get("answer_leakage"))
            and bool(pred_search.strip())
            and bool(cf.get("search_cf_valid"))
        )
        edge = edge_record(real_score, cf_scores, gate_pass)
        old_total = float(row.get("DAGIG_total") or row.get("DAGIG_v2_total") or 0.0)
        old_search = float(row.get("R_search") or 0.0)
        dagig_v3_total = old_total - old_search + float(edge["edge_credit"])
        edges = dict(row.get("edges") or {})
        edges["search_query_to_retrieved_evidence_v3"] = edge
        out_rows.append(
            {
                **row,
                "split": split,
                "R_search_old": old_search,
                "R_search_v3": float(edge["edge_credit"]),
                "DAGIG_v3_total": float(dagig_v3_total),
                "search_cf_v3_valid": bool(cf.get("search_cf_valid")),
                "search_cf_v3_reason": cf.get("reason"),
                "search_cf_v3_rank_delta": cf.get("rank_delta"),
                "search_cf_v3_support_drop": cf.get("support_drop"),
                "search_cf_v3_real_beats_best_valid_cf": cf.get("real_beats_best_valid_cf"),
                "supported_answer": supported_answer(row),
                "edges": edges,
            }
        )
    write_jsonl(out_dir / f"scored_rollouts_{split}.jsonl", out_rows)
    return out_rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        vals = [row.get(key) for row in rows if row.get(key) is not None]
        if not vals:
            return None
        return float(mean(float(v) for v in vals))

    return {
        "n": len(rows),
        "DAGIG_v3_total": avg("DAGIG_v3_total"),
        "R_search_v3": avg("R_search_v3"),
        "R_search_old": avg("R_search_old"),
        "retrieval_r5": avg("retrieval_r5"),
        "retrieval_mrr": avg("retrieval_mrr"),
        "evidence_support": avg("evidence_support"),
        "supported_answer": avg("supported_answer"),
        "answer_f1": avg("answer_f1"),
        "search_cf_v3_valid": avg("search_cf_v3_valid"),
    }


def main() -> int:
    args = parse_args()
    input_scored_dir = Path(args.input_scored_dir)
    search_cf_dir = Path(args.search_cf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split in args.splits:
        rows = score_split(split, input_scored_dir, search_cf_dir, out_dir)
        summaries.append({"split": split, **aggregate(rows)})
    write_json(out_dir / "search_cf_v3_score_summary.json", summaries)
    display = [{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()} for row in summaries]
    (out_dir / "search_cf_v3_score_summary.md").write_text(
        "# DAG-IG Search CF v3 Score Summary\n\n" + md_table(display) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
