#!/usr/bin/env python3
"""Counterfactual edge-level DAG-IG scorer for grounded rollouts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from grounded_pipeline_utils import (
    DINO_CONFIG,
    DINO_WEIGHTS,
    bbox_iou,
    center_hit,
    extract_segments,
    load_jsonl,
    md_table,
    token_f1,
    tokenize,
    write_csv,
    write_json,
    write_jsonl,
)


SCORER_PATH = Path(__file__).with_name("26_score_grounded_rollouts.py")
SPEC = importlib.util.spec_from_file_location("grounded_rollout_scorer", SCORER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import scorer from {SCORER_PATH}")
SCORER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER_MODULE)
GroundedRewardScorer = SCORER_MODULE.GroundedRewardScorer
query_anchor_hit = SCORER_MODULE.query_anchor_hit
query_specificity_score = SCORER_MODULE.query_specificity_score
answer_leaks = SCORER_MODULE.answer_leaks
contains_match = SCORER_MODULE.contains_match


TAGS = ["ground", "observe", "search_decision", "search", "evidence", "answer"]
EDGES = [
    "ground_expression_to_dino_crop",
    "crop_to_observation",
    "observation_to_search_query",
    "search_query_to_retrieved_evidence",
    "evidence_to_answer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl_data_jsonl", default="data/dagig_rn03_10_grounded_rl/grounded_rl_dev.jsonl")
    parser.add_argument("--counterfactual_jsonl", default="data/dagig_rn03_10_counterfactuals/counterfactual_dev.jsonl")
    parser.add_argument("--rollouts_jsonl", default="")
    parser.add_argument("--output_jsonl", default="results/dagig_rn03_10_counterfactual_dagig/scored/dev.jsonl")
    parser.add_argument("--summary_json", default="results/dagig_rn03_10_counterfactual_dagig/scored/dev_summary.json")
    parser.add_argument("--summary_md", default="")
    parser.add_argument("--corpus_jsonl", default="data/dagig_rn03_10_retrieval/corpus.jsonl")
    parser.add_argument("--targets_json", default="data/dagig_rn03_10_retrieval/targets.json")
    parser.add_argument("--package_root", default="data/dagig_rn03_10_ground_expr_v3_full")
    parser.add_argument("--grounding_config", default=DINO_CONFIG)
    parser.add_argument("--grounding_weights", default=DINO_WEIGHTS)
    parser.add_argument("--box_threshold", type=float, default=0.10)
    parser.add_argument("--text_threshold", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--use_target_rollouts", action="store_true")
    parser.add_argument("--no_dino", action="store_true")
    return parser.parse_args()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_delta(real: float, cf: float) -> float:
    denom = max(1.0, abs(real), abs(cf))
    return (real - cf) / denom


def normalize_text(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def has_answer_leak(example: dict[str, Any], prediction: str) -> bool:
    answer = str(example.get("gold_answer") or example.get("answer") or "")
    segments = extract_segments(prediction, TAGS)
    leak_text = "\n".join(str(segments.get(tag, "")) for tag in ["ground", "observe", "search_decision", "search"])
    return answer_leaks(answer, leak_text)


def edge_record(real_score: float, cf_score: float, gate_pass: bool) -> dict[str, Any]:
    delta = real_score - cf_score
    norm = normalize_delta(real_score, cf_score)
    return {
        "real_score": float(real_score),
        "counterfactual_score": float(cf_score),
        "delta": float(delta),
        "normalized_delta": float(norm),
        "gate_pass": bool(gate_pass),
        "edge_credit": float(norm if gate_pass else min(norm, 0.0)),
    }


def doc_supports_answer(text: str, answer: str) -> bool:
    ans = normalize_text(answer)
    body = normalize_text(text)
    return bool(ans and len(ans.replace(" ", "")) >= 2 and ans in body)


class CounterfactualDAGIGScorer:
    def __init__(
        self,
        corpus_jsonl: str,
        targets_json: str,
        package_root: str,
        grounding_config: str = DINO_CONFIG,
        grounding_weights: str = DINO_WEIGHTS,
        box_threshold: float = 0.10,
        text_threshold: float = 0.10,
        device: str = "cuda",
        use_dino: bool = True,
    ):
        self.base = GroundedRewardScorer(
            corpus_jsonl=corpus_jsonl,
            targets_json=targets_json,
            package_root=package_root,
            grounding_config=grounding_config,
            grounding_weights=grounding_weights,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
            use_dino=use_dino,
        )
        self.ground_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def grounding_score(self, example: dict[str, Any], expression: str) -> dict[str, Any]:
        key = (str(example.get("sample_id")), expression)
        if key not in self.ground_cache:
            self.ground_cache[key] = self.base.score_grounding(example, expression)
        return self.ground_cache[key]

    @staticmethod
    def ground_quality(metrics: dict[str, Any]) -> float:
        if metrics.get("ground_detected") is None:
            return 0.0
        iou = float(metrics.get("ground_iou") or 0.0)
        return clamp01(iou + (0.2 if metrics.get("ground_center_hit") else 0.0) + (0.1 if metrics.get("ground_detected") else 0.0))

    @staticmethod
    def crop_quality(cf_box: list[float] | None, gold_box: list[float] | None) -> float:
        if not cf_box or not gold_box:
            return 0.0
        iou = bbox_iou(cf_box, gold_box)
        hit = center_hit(cf_box, gold_box)
        image_like = iou > 0.95 and cf_box[0] <= 1 and cf_box[1] <= 1
        return clamp01(iou + (0.15 if hit else 0.0) + (0.1 if image_like else 0.0))

    def retrieval_score(self, example: dict[str, Any], query: str) -> dict[str, Any]:
        ret = self.base.retrieval_metrics(example, query)
        r5 = 1.0 if ret.get("retrieval_r5") is True else 0.0
        r1 = 1.0 if ret.get("retrieval_r1") is True else 0.0
        mrr = float(ret.get("retrieval_mrr") or 0.0)
        ret["retrieval_score"] = clamp01(0.45 * r5 + 0.25 * r1 + 0.30 * mrr)
        return ret

    def score(self, example: dict[str, Any], cf: dict[str, Any], prediction: str) -> dict[str, Any]:
        segments = extract_segments(prediction, TAGS)
        target = example.get("target_segments") if isinstance(example.get("target_segments"), dict) else {}
        anchor = str(example.get("semantic_anchor") or example.get("visual_anchor") or example.get("teacher_ground_expression") or "")
        answer = str(example.get("gold_answer") or example.get("answer") or target.get("answer") or "")
        teacher_observe = str(example.get("teacher_observation") or target.get("observe") or "")
        teacher_evidence = str(example.get("teacher_evidence_text") or target.get("evidence") or example.get("evidence") or "")
        pred_ground = segments.get("ground", "")
        pred_observe = segments.get("observe", "")
        pred_query = segments.get("search", "")
        pred_evidence = segments.get("evidence", "")
        pred_answer = segments.get("answer", "")

        base_metrics = self.base.score(example, prediction, "dagig_grounded")
        format_valid = bool(base_metrics.get("format_valid"))
        no_leak = not has_answer_leak(example, prediction)

        real_ground = self.grounding_score(example, pred_ground)
        real_ground_score = self.ground_quality(real_ground)
        ground_cf_scores = [
            self.ground_quality(self.grounding_score(example, str(item.get("text", ""))))
            for item in cf.get("counterfactuals", {}).get("ground_cf", [])
        ]
        cf_ground_score = max(ground_cf_scores or [0.0])
        edge_ground = edge_record(real_ground_score, cf_ground_score, format_valid and bool(pred_ground.strip()))

        gold_box = cf.get("gold_bbox_pixel_xyxy")
        observe_real = clamp01(0.55 * token_f1(pred_observe, teacher_observe) + 0.30 * float(query_anchor_hit(pred_observe, anchor)) + 0.15 * real_ground_score)
        crop_cf_scores = [
            self.crop_quality(item.get("bbox_pixel_xyxy"), gold_box)
            for item in cf.get("counterfactuals", {}).get("crop_cf", [])
        ]
        observe_cf_supported = max(crop_cf_scores or [0.0]) * observe_real
        edge_observe = edge_record(observe_real, observe_cf_supported, edge_ground["edge_credit"] > 0 and bool(pred_observe.strip()))

        query_real = clamp01(
            0.40 * float(query_anchor_hit(pred_query, anchor))
            + 0.25 * token_f1(pred_query, pred_observe)
            + 0.20 * query_specificity_score(pred_query)
            + 0.15 * token_f1(pred_query, str(example.get("teacher_search_query") or target.get("search") or ""))
        )
        observe_cf_scores = []
        for item in cf.get("counterfactuals", {}).get("observe_cf", []):
            cf_obs = str(item.get("text", ""))
            observe_cf_scores.append(
                clamp01(0.45 * token_f1(pred_query, cf_obs) + 0.25 * float(query_anchor_hit(pred_query, cf_obs)) + 0.30 * query_specificity_score(pred_query))
            )
        edge_search = edge_record(query_real, max(observe_cf_scores or [0.0]), edge_observe["edge_credit"] > 0 and bool(pred_query.strip()))

        retrieval_real = self.retrieval_score(example, pred_query)
        search_cf_scores = [self.retrieval_score(example, str(item.get("text", ""))).get("retrieval_score", 0.0) for item in cf.get("counterfactuals", {}).get("search_cf", [])]
        edge_evidence = edge_record(
            float(retrieval_real.get("retrieval_score") or 0.0),
            max(float(x or 0.0) for x in (search_cf_scores or [0.0])),
            edge_search["edge_credit"] > 0,
        )

        evidence_support_real = clamp01(0.50 * token_f1(pred_evidence, teacher_evidence) + 0.30 * float(base_metrics.get("evidence_support")) + 0.20 * float(retrieval_real.get("retrieval_r5") is True))
        answer_quality = 1.0 if contains_match(pred_answer, answer) else 0.5 * token_f1(pred_answer, answer)
        real_answer_score = clamp01(answer_quality * evidence_support_real)
        cf_answer_scores = []
        for item in cf.get("counterfactuals", {}).get("evidence_cf", []):
            text = str(item.get("text", ""))
            cf_support = clamp01(0.7 * float(doc_supports_answer(text, answer)) + 0.3 * token_f1(text, pred_evidence))
            cf_answer_scores.append(answer_quality * cf_support)
        edge_answer = edge_record(real_answer_score, max(cf_answer_scores or [0.0]), edge_evidence["edge_credit"] > 0 and bool(pred_answer.strip()))

        r_cost = 0.0
        if not format_valid:
            r_cost -= 0.2
        if not no_leak:
            r_cost -= 0.3
        if base_metrics.get("unsupported_answer"):
            r_cost -= 0.2
        if base_metrics.get("spurious_success"):
            r_cost -= 0.2

        edges = {
            "ground_expression_to_dino_crop": edge_ground,
            "crop_to_observation": edge_observe,
            "observation_to_search_query": edge_search,
            "search_query_to_retrieved_evidence": edge_evidence,
            "evidence_to_answer": edge_answer,
        }
        components = {
            "R_ground": edge_ground["edge_credit"],
            "R_observe": edge_observe["edge_credit"],
            "R_search": edge_search["edge_credit"],
            "R_evidence": edge_evidence["edge_credit"],
            "R_answer": edge_answer["edge_credit"],
            "R_cost": r_cost,
        }
        total = sum(float(v) for v in components.values())
        return {
            "DAGIG_total": float(total),
            **components,
            "edges": edges,
            "format_valid": format_valid,
            "answer_leakage": not no_leak,
            "prediction_segments": segments,
            "ground_iou": real_ground.get("ground_iou"),
            "ground_center_hit": real_ground.get("ground_center_hit"),
            "ground_detected": real_ground.get("ground_detected"),
            "retrieval_r1": retrieval_real.get("retrieval_r1"),
            "retrieval_r5": retrieval_real.get("retrieval_r5"),
            "retrieval_mrr": retrieval_real.get("retrieval_mrr"),
            "query_anchor_hit": base_metrics.get("query_anchor_hit"),
            "query_specificity": base_metrics.get("query_specificity"),
            "evidence_support": base_metrics.get("evidence_support"),
            "unsupported_answer": base_metrics.get("unsupported_answer"),
            "spurious_success": base_metrics.get("spurious_success"),
            "answer_em": base_metrics.get("answer_em"),
            "answer_f1": base_metrics.get("answer_f1"),
            "base_dagig_heuristic_reward": base_metrics.get("reward_total"),
        }


def load_rollout_predictions(path: str | Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    out = []
    for row in rows:
        if isinstance(row.get("rollouts"), list):
            for idx, rollout in enumerate(row["rollouts"]):
                out.append(
                    {
                        "sample_id": row.get("sample_id") or rollout.get("sample_id"),
                        "rollout_index": rollout.get("rollout_index", idx),
                        "prediction": rollout.get("prediction") or rollout.get("completion") or rollout.get("text") or "",
                    }
                )
        else:
            out.append(
                {
                    "sample_id": row.get("sample_id"),
                    "rollout_index": row.get("rollout_index", 0),
                    "prediction": row.get("prediction") or row.get("completion") or row.get("text") or "",
                }
            )
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}

    def avg(key: str) -> float | None:
        vals = [row.get(key) for row in rows if row.get(key) is not None]
        if not vals:
            return None
        return float(mean(float(v) for v in vals))

    return {
        "n": len(rows),
        "DAGIG_total": avg("DAGIG_total"),
        "R_ground": avg("R_ground"),
        "R_observe": avg("R_observe"),
        "R_search": avg("R_search"),
        "R_evidence": avg("R_evidence"),
        "R_answer": avg("R_answer"),
        "R_cost": avg("R_cost"),
        "format_valid": avg("format_valid"),
        "ground_iou": avg("ground_iou"),
        "center_hit": avg("ground_center_hit"),
        "retrieval_r5": avg("retrieval_r5"),
        "evidence_support": avg("evidence_support"),
        "answer_f1": avg("answer_f1"),
    }


def main() -> int:
    args = parse_args()
    examples = load_jsonl(args.rl_data_jsonl, limit=args.limit)
    cfs = load_jsonl(args.counterfactual_jsonl, limit=args.limit)
    by_id = {str(row.get("sample_id")): row for row in examples}
    cf_by_id = {str(row.get("sample_id")): row for row in cfs}
    if args.use_target_rollouts:
        rollout_rows = [
            {"sample_id": row.get("sample_id"), "rollout_index": 0, "prediction": row.get("model_target_text", "")}
            for row in examples
        ]
    elif args.rollouts_jsonl:
        rollout_rows = load_rollout_predictions(args.rollouts_jsonl)
        if args.limit:
            rollout_rows = rollout_rows[: args.limit]
    else:
        raise ValueError("Pass --use_target_rollouts or --rollouts_jsonl")

    scorer = CounterfactualDAGIGScorer(
        corpus_jsonl=args.corpus_jsonl,
        targets_json=args.targets_json,
        package_root=args.package_root,
        grounding_config=args.grounding_config,
        grounding_weights=args.grounding_weights,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
        use_dino=not args.no_dino,
    )
    scored = []
    for row in rollout_rows:
        sample_id = str(row.get("sample_id") or "")
        example = by_id.get(sample_id)
        cf = cf_by_id.get(sample_id)
        if example is None or cf is None:
            continue
        score = scorer.score(example, cf, str(row.get("prediction") or ""))
        scored.append({"sample_id": sample_id, "rollout_index": row.get("rollout_index", 0), "prediction": row.get("prediction"), **score})
    summary = aggregate(scored)
    write_jsonl(args.output_jsonl, scored)
    write_json(args.summary_json, summary)
    if args.summary_md:
        rows = [{"metric": k, "value": v} for k, v in summary.items()]
        Path(args.summary_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_md).write_text("# Counterfactual DAG-IG Summary\n\n" + md_table(rows) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
