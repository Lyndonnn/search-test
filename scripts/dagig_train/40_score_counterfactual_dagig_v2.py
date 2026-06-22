#!/usr/bin/env python3
"""Score rollouts with DAG-IG v2 typed counterfactual edge credit."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
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

VERIFIER_PATH = Path(__file__).with_name("39_evidence_support_verifier.py")
VERIFIER_SPEC = importlib.util.spec_from_file_location("dagig_v2_evidence_verifier", VERIFIER_PATH)
if VERIFIER_SPEC is None or VERIFIER_SPEC.loader is None:
    raise ImportError(f"Could not import verifier from {VERIFIER_PATH}")
VERIFIER_MODULE = importlib.util.module_from_spec(VERIFIER_SPEC)
VERIFIER_SPEC.loader.exec_module(VERIFIER_MODULE)
verify_evidence = VERIFIER_MODULE.verify_evidence


TAGS = ["ground", "observe", "search_decision", "search", "evidence", "answer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl_data_jsonl", required=True)
    parser.add_argument("--counterfactual_jsonl", required=True)
    parser.add_argument("--search_counterfactual_jsonl", required=True)
    parser.add_argument("--rollouts_jsonl", required=True)
    parser.add_argument("--audit_csv", default="")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
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
    parser.add_argument("--no_dino", action="store_true")
    parser.add_argument("--verifier_mode", choices=["heuristic", "hybrid", "local_llm"], default="hybrid")
    return parser.parse_args()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_delta(real: float, cf: float) -> float:
    return (float(real) - float(cf)) / max(1.0, abs(float(real)), abs(float(cf)))


def norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def has_answer_leak(example: dict[str, Any], prediction: str) -> bool:
    answer = str(example.get("gold_answer") or example.get("answer") or "")
    segments = extract_segments(prediction, TAGS)
    leak_text = "\n".join(str(segments.get(tag, "")) for tag in ["ground", "observe", "search_decision", "search"])
    return answer_leaks(answer, leak_text)


def edge_record(real_score: float, cf_scores: list[float], gate_pass: bool) -> dict[str, Any]:
    if not cf_scores:
        cf_scores = [0.0]
    cf_best = max(float(v) for v in cf_scores)
    cf_mean = mean(float(v) for v in cf_scores)
    delta_best = normalize_delta(real_score, cf_best)
    delta_mean = normalize_delta(real_score, cf_mean)
    credit = delta_mean if gate_pass else min(delta_mean, 0.0)
    return {
        "real_score": float(real_score),
        "cf_best_score": float(cf_best),
        "cf_mean_score": float(cf_mean),
        "delta_best": float(delta_best),
        "delta_mean": float(delta_mean),
        "edge_credit": float(credit),
        "gate_pass": bool(gate_pass),
    }


def load_failed_sample_ids(path: str) -> set[str]:
    if not path or not Path(path).is_file():
        return set()
    failed: set[str] = set()
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("overall_cf_quality") == "fail":
                failed.add(str(row.get("sample_id")))
    return failed


def load_rollouts(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = load_jsonl(path, limit=limit)
    out = []
    for row in rows:
        out.append(
            {
                "sample_id": row.get("sample_id"),
                "split": row.get("split"),
                "rollout_index": row.get("rollout_index", 0),
                "prediction": row.get("prediction") or row.get("completion") or row.get("text") or "",
            }
        )
    return out


class DAGIGV2Scorer:
    def __init__(
        self,
        corpus_jsonl: str,
        targets_json: str,
        package_root: str,
        grounding_config: str,
        grounding_weights: str,
        box_threshold: float,
        text_threshold: float,
        device: str,
        use_dino: bool,
        verifier_mode: str,
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
        self.verifier_mode = verifier_mode
        self.ground_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def grounding_metrics(self, example: dict[str, Any], expression: str) -> dict[str, Any]:
        key = (str(example.get("sample_id")), str(expression or ""))
        if key not in self.ground_cache:
            self.ground_cache[key] = self.base.score_grounding(example, expression)
        return self.ground_cache[key]

    @staticmethod
    def ground_quality(metrics: dict[str, Any]) -> float:
        if metrics.get("ground_detected") is None:
            return 0.0
        iou = float(metrics.get("ground_iou") or 0.0)
        return clamp01(0.70 * iou + 0.20 * float(metrics.get("ground_center_hit") is True) + 0.10 * float(metrics.get("ground_detected") is True))

    @staticmethod
    def crop_quality(box: list[float] | None, gold: list[float] | None) -> float:
        if not box or not gold:
            return 0.0
        return clamp01(0.80 * bbox_iou(box, gold) + 0.20 * float(center_hit(box, gold)))

    def retrieval_metrics(self, example: dict[str, Any], query: str) -> dict[str, Any]:
        ret = self.base.retrieval_metrics(example, query)
        r5 = 1.0 if ret.get("retrieval_r5") is True else 0.0
        r1 = 1.0 if ret.get("retrieval_r1") is True else 0.0
        mrr = float(ret.get("retrieval_mrr") or 0.0)
        ret["retrieval_score"] = clamp01(0.45 * r5 + 0.25 * r1 + 0.30 * mrr)
        return ret

    def support(self, example: dict[str, Any], answer: str, evidence: str) -> dict[str, Any]:
        return verify_evidence(
            example.get("question"),
            answer,
            evidence,
            example.get("gold_answer") or example.get("answer"),
            self.verifier_mode,
        )

    def score(self, example: dict[str, Any], cf: dict[str, Any], search_cf: dict[str, Any], prediction: str) -> dict[str, Any]:
        segments = extract_segments(prediction, TAGS)
        target = example.get("target_segments") if isinstance(example.get("target_segments"), dict) else {}
        answer_gold = str(example.get("gold_answer") or example.get("answer") or target.get("answer") or "")
        anchor = str(example.get("semantic_anchor") or example.get("visual_anchor") or example.get("teacher_ground_expression") or "")
        teacher_observe = str(example.get("teacher_observation") or target.get("observe") or "")
        teacher_query = str(example.get("teacher_search_query") or target.get("search") or "")
        pred_ground = segments.get("ground", "")
        pred_observe = segments.get("observe", "")
        pred_query = segments.get("search", "")
        pred_evidence = segments.get("evidence", "")
        pred_answer = segments.get("answer", "")

        format_valid = all(bool(segments.get(tag, "").strip()) for tag in TAGS)
        answer_leakage = has_answer_leak(example, prediction)
        answer_em = 1.0 if contains_match(pred_answer, answer_gold) else 0.0
        answer_f1 = token_f1(pred_answer, answer_gold)

        real_ground = self.grounding_metrics(example, pred_ground)
        real_ground_score = self.ground_quality(real_ground)
        ground_cf_scores = [
            self.ground_quality(self.grounding_metrics(example, str(item.get("text") or "")))
            for item in (cf.get("counterfactuals", {}).get("ground_cf") or [])
        ]
        edge_ground = edge_record(real_ground_score, ground_cf_scores, format_valid and bool(pred_ground.strip()))

        gold_box = cf.get("gold_bbox_pixel_xyxy")
        observe_real = clamp01(
            0.45 * token_f1(pred_observe, teacher_observe)
            + 0.25 * float(query_anchor_hit(pred_observe, anchor))
            + 0.30 * real_ground_score
        )
        crop_cf_scores = [
            self.crop_quality(item.get("bbox_pixel_xyxy"), gold_box)
            for item in (cf.get("counterfactuals", {}).get("crop_cf") or [])
        ]
        observe_cf_scores = [observe_real * score for score in crop_cf_scores]
        grounding_ok = bool(real_ground.get("ground_center_hit")) or float(real_ground.get("ground_iou") or 0.0) >= 0.10
        edge_observe = edge_record(observe_real, observe_cf_scores, edge_ground["edge_credit"] > -0.05 and grounding_ok and bool(pred_observe.strip()))

        query_dependency_real = clamp01(
            0.35 * float(query_anchor_hit(pred_query, anchor))
            + 0.25 * token_f1(pred_query, pred_observe)
            + 0.25 * token_f1(pred_query, teacher_query)
            + 0.15 * query_specificity_score(pred_query)
        )
        obs_cf_scores = []
        for item in (cf.get("counterfactuals", {}).get("observe_cf") or []):
            obs = str(item.get("text") or "")
            obs_cf_scores.append(
                clamp01(
                    0.35 * float(query_anchor_hit(pred_query, obs))
                    + 0.25 * token_f1(pred_query, obs)
                    + 0.25 * token_f1(pred_query, teacher_query)
                    + 0.15 * query_specificity_score(pred_query)
                )
            )
        edge_query_dependency = edge_record(query_dependency_real, obs_cf_scores, edge_observe["edge_credit"] > -0.05 and bool(pred_query.strip()))

        retrieval_real = self.retrieval_metrics(example, pred_query)
        search_cf_scores = []
        for item in search_cf.get("queries", []):
            if item.get("type") == "real":
                continue
            search_cf_scores.append(float(item.get("retrieval_score") or 0.0))
        if not search_cf_scores:
            for item in (cf.get("counterfactuals", {}).get("search_cf") or []):
                search_cf_scores.append(float(self.retrieval_metrics(example, str(item.get("text") or "")).get("retrieval_score") or 0.0))
        edge_search = edge_record(
            float(retrieval_real.get("retrieval_score") or 0.0),
            search_cf_scores,
            edge_query_dependency["edge_credit"] > -0.10 and bool(pred_query.strip()),
        )

        support_real = self.support(example, pred_answer or answer_gold, pred_evidence)
        evidence_real_score = float(support_real.get("support_score") or 0.0)
        evidence_cf_scores = []
        for item in (cf.get("counterfactuals", {}).get("evidence_cf") or []):
            evidence_cf_scores.append(float(self.support(example, pred_answer or answer_gold, str(item.get("text") or "")).get("support_score") or 0.0))
        retrieval_supported = retrieval_real.get("retrieval_r5") is True
        edge_evidence = edge_record(
            evidence_real_score,
            evidence_cf_scores,
            edge_search["edge_credit"] > -0.05 and (retrieval_supported or evidence_real_score >= 0.40),
        )

        answer_quality = max(answer_em, 0.5 * answer_f1)
        real_answer_score = clamp01(answer_quality * evidence_real_score)
        cf_answer_scores = [clamp01(answer_quality * score) for score in evidence_cf_scores]
        supported_answer = bool(support_real.get("supports_answer") and answer_quality > 0.0)
        edge_answer = edge_record(real_answer_score, cf_answer_scores, edge_evidence["edge_credit"] > -0.05 and bool(pred_answer.strip()))

        unsupported_answer = bool(pred_answer.strip() and answer_em and not support_real.get("supports_answer"))
        spurious_success = bool(answer_em and (not support_real.get("supports_answer") or retrieval_real.get("retrieval_r5") is not True))
        r_cost = 0.0
        if not format_valid:
            r_cost -= 0.20
        if answer_leakage:
            r_cost -= 0.30
        if unsupported_answer:
            r_cost -= 0.20
        if spurious_success:
            r_cost -= 0.20

        components = {
            "R_ground": edge_ground["edge_credit"],
            "R_observe": edge_observe["edge_credit"],
            "R_query_dependency": edge_query_dependency["edge_credit"],
            "R_search": edge_search["edge_credit"],
            "R_evidence": edge_evidence["edge_credit"],
            "R_answer": edge_answer["edge_credit"],
            "R_cost": r_cost,
        }
        total = (
            components["R_ground"]
            + 0.50 * components["R_observe"]
            + 0.50 * components["R_query_dependency"]
            + 1.25 * components["R_search"]
            + 1.25 * components["R_evidence"]
            + components["R_answer"]
            + components["R_cost"]
        )
        return {
            "DAGIG_v2_total": float(total),
            **{key: float(value) for key, value in components.items()},
            "edges": {
                "ground_expression_to_dino_crop": edge_ground,
                "crop_to_observation": edge_observe,
                "observation_to_search_query": edge_query_dependency,
                "search_query_to_retrieved_evidence": edge_search,
                "evidence_to_answer": edge_answer,
            },
            "format_valid": bool(format_valid),
            "answer_leakage": bool(answer_leakage),
            "ground_iou": real_ground.get("ground_iou"),
            "ground_center_hit": real_ground.get("ground_center_hit"),
            "ground_detected": real_ground.get("ground_detected"),
            "retrieval_r1": retrieval_real.get("retrieval_r1"),
            "retrieval_r5": retrieval_real.get("retrieval_r5"),
            "retrieval_mrr": retrieval_real.get("retrieval_mrr"),
            "retrieval_score": retrieval_real.get("retrieval_score"),
            "query_anchor_hit": query_anchor_hit(pred_query, anchor),
            "query_specificity": query_specificity_score(pred_query),
            "evidence_support": bool(support_real.get("supports_answer")),
            "evidence_support_score": evidence_real_score,
            "supported_answer": bool(supported_answer),
            "unsupported_answer": bool(unsupported_answer),
            "spurious_success": bool(spurious_success),
            "answer_em": float(answer_em),
            "answer_f1": float(answer_f1),
            "prediction_segments": segments,
            "support_verifier": support_real,
        }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        vals = [row.get(key) for row in rows if row.get(key) is not None]
        if not vals:
            return None
        return float(mean(float(v) for v in vals))

    return {
        "n": len(rows),
        "DAGIG_v2_total": avg("DAGIG_v2_total"),
        "R_ground": avg("R_ground"),
        "R_search": avg("R_search"),
        "R_evidence": avg("R_evidence"),
        "R_answer": avg("R_answer"),
        "ground_iou": avg("ground_iou"),
        "center_hit": avg("ground_center_hit"),
        "retrieval_r5": avg("retrieval_r5"),
        "evidence_support": avg("evidence_support"),
        "supported_answer": avg("supported_answer"),
        "answer_f1": avg("answer_f1"),
    }


def main() -> int:
    args = parse_args()
    examples = {str(row.get("sample_id")): row for row in load_jsonl(args.rl_data_jsonl)}
    cfs = {str(row.get("sample_id")): row for row in load_jsonl(args.counterfactual_jsonl)}
    search_cfs = {str(row.get("sample_id")): row for row in load_jsonl(args.search_counterfactual_jsonl)}
    failed = load_failed_sample_ids(args.audit_csv)
    rollouts = load_rollouts(args.rollouts_jsonl, limit=args.limit)
    scorer = DAGIGV2Scorer(
        corpus_jsonl=args.corpus_jsonl,
        targets_json=args.targets_json,
        package_root=args.package_root,
        grounding_config=args.grounding_config,
        grounding_weights=args.grounding_weights,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
        use_dino=not args.no_dino,
        verifier_mode=args.verifier_mode,
    )
    scored = []
    skipped_quality = 0
    for row in rollouts:
        sample_id = str(row.get("sample_id") or "")
        if sample_id in failed:
            skipped_quality += 1
            continue
        example = examples.get(sample_id)
        cf = cfs.get(sample_id)
        search_cf = search_cfs.get(sample_id, {})
        if example is None or cf is None:
            continue
        score = scorer.score(example, cf, search_cf, str(row.get("prediction") or ""))
        scored.append(
            {
                "sample_id": sample_id,
                "split": example.get("split"),
                "rollout_index": row.get("rollout_index", 0),
                "prediction": row.get("prediction"),
                **score,
            }
        )
    summary = aggregate(scored) | {"skipped_cf_quality_fail": skipped_quality}
    write_jsonl(args.output_jsonl, scored)
    write_json(args.summary_json, summary)
    if args.summary_md:
        rows = [{"metric": key, "value": value} for key, value in summary.items()]
        Path(args.summary_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_md).write_text("# DAG-IG v2 Scored Rollout Summary\n\n" + md_table(rows) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
