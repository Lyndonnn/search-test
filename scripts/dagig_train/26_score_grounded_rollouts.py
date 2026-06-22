#!/usr/bin/env python3
"""Score grounded rollouts under outcome, generic process, and DAG-IG rewards."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from grounded_pipeline_utils import (
    DINO_CONFIG,
    DINO_WEIGHTS,
    bbox_iou,
    center_hit,
    extract_segments,
    get_gold_bbox,
    image_size,
    is_extreme_box,
    load_groundingdino_model,
    load_jsonl,
    normalize_bbox,
    predict_groundingdino,
    row_image_path,
    token_f1,
    tokenize,
    write_json,
    write_jsonl,
)


TAGS = ["ground", "observe", "search_decision", "search", "evidence", "answer"]
REWARD_MODES = ["outcome_only", "outcome_plus_ground_penalty", "generic_process", "dagig_grounded"]
FORBIDDEN_RE = re.compile(r"\b(?:bbox|bounding box|red\s*box|red-box|annotation|annotated)\b", re.I)
GENERIC_GROUND_WORDS = {
    "area",
    "image",
    "object",
    "region",
    "thing",
    "item",
    "part",
    "place",
    "building",
    "sign",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl_data_jsonl", default="data/dagig_rn03_10_grounded_rl/grounded_rl_dev.jsonl")
    parser.add_argument("--rollouts_jsonl", default="")
    parser.add_argument("--output_jsonl", default="results/dagig_rn03_10_grounded_rl/reward_smoke/scored_rollouts.jsonl")
    parser.add_argument("--summary_json", default="results/dagig_rn03_10_grounded_rl/reward_smoke/summary.json")
    parser.add_argument("--reward_mode", choices=REWARD_MODES, default="dagig_grounded")
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


def normalize(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def contains_match(pred: Any, target: Any) -> bool:
    pred_norm = normalize(pred)
    target_norm = normalize(target)
    return bool(pred_norm and target_norm and (pred_norm in target_norm or target_norm in pred_norm))


def answer_leaks(answer: Any, text: str) -> bool:
    ans = normalize(answer)
    body = normalize(text)
    if not ans or not body:
        return False
    compact = ans.replace(" ", "")
    min_len = 2 if compact.isdigit() else 3
    return len(compact) >= min_len and ans in body


def query_anchor_hit(query: str, anchor: str) -> bool:
    q_tokens = set(tokenize(query))
    anchor_tokens = [tok for tok in tokenize(anchor) if len(tok) > 1]
    return bool(anchor_tokens and any(tok in q_tokens for tok in anchor_tokens))


def query_specificity_score(query: str) -> float:
    toks = tokenize(query)
    if not toks:
        return 0.0
    length_score = min(len(toks) / 8.0, 1.0)
    content_score = sum(1 for tok in toks if len(tok) >= 5 or tok.isdigit()) / max(1, len(toks))
    return min(1.0, 0.65 * length_score + 0.35 * min(1.0, content_score * 2.0))


def too_generic_ground(text: str) -> bool:
    toks = [tok for tok in tokenize(text) if len(tok) > 1]
    if len(toks) < 3:
        return True
    return len(set(toks) - GENERIC_GROUND_WORDS) <= 1


class BM25:
    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = docs
        self.doc_tokens = [tokenize(str(doc.get("text", ""))) for doc in docs]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        n_docs = len(docs)
        self.idf = {tok: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5)) for tok, freq in df.items()}

    def rank(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        q_counts = Counter(tokenize(query))
        ranked = []
        k1 = 1.2
        b = 0.75
        for doc, toks, dl in zip(self.docs, self.doc_tokens, self.doc_lens):
            tf = Counter(toks)
            score = 0.0
            for tok, qf in q_counts.items():
                if tok not in tf:
                    continue
                denom = tf[tok] + k1 * (1 - b + b * dl / max(self.avgdl, 1e-6))
                score += self.idf.get(tok, 0.0) * (tf[tok] * (k1 + 1) / denom) * qf
            ranked.append((str(doc["doc_id"]), score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:k]


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    docs = []
    for idx, row in enumerate(load_jsonl(path)):
        text = str(row.get("text") or row.get("contents") or row.get("content") or "")
        doc_id = str(row.get("doc_id") or row.get("id") or idx)
        if text:
            docs.append({"doc_id": doc_id, "text": text, "sample_id": row.get("sample_id")})
    return docs


def load_targets(path: str | Path) -> dict[str, list[str]]:
    if not Path(path).is_file():
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    return {str(k): [str(x) for x in v] for k, v in payload.items() if isinstance(v, list)}


def reciprocal_rank(ranked: list[tuple[str, float]], targets: set[str]) -> float | None:
    if not targets:
        return None
    for idx, (doc_id, _score) in enumerate(ranked, start=1):
        if doc_id in targets:
            return 1.0 / idx
    return 0.0


class GroundedRewardScorer:
    def __init__(
        self,
        corpus_jsonl: str = "data/dagig_rn03_10_retrieval/corpus.jsonl",
        targets_json: str = "data/dagig_rn03_10_retrieval/targets.json",
        package_root: str = "data/dagig_rn03_10_ground_expr_v3_full",
        grounding_config: str = DINO_CONFIG,
        grounding_weights: str = DINO_WEIGHTS,
        box_threshold: float = 0.10,
        text_threshold: float = 0.10,
        device: str = "cuda",
        use_dino: bool = True,
    ):
        self.package_root = package_root
        self.grounding_config = grounding_config
        self.grounding_weights = grounding_weights
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.corpus = load_corpus(corpus_jsonl)
        self.bm25 = BM25(self.corpus) if self.corpus else None
        self.targets = load_targets(targets_json)
        self.use_dino = use_dino
        self._dino = None

    @property
    def dino(self) -> Any:
        if self._dino is None:
            self._dino = load_groundingdino_model(self.grounding_config, self.grounding_weights, self.device)
        return self._dino

    def score_grounding(self, example: dict[str, Any], ground_text: str) -> dict[str, Any]:
        if not self.use_dino:
            return {
                "ground_detected": None,
                "ground_bbox": None,
                "ground_score": None,
                "ground_iou": None,
                "ground_center_hit": None,
                "ground_extreme_box": None,
                "ground_num_detections": None,
            }
        if not ground_text.strip():
            return {
                "ground_detected": False,
                "ground_bbox": None,
                "ground_score": None,
                "ground_iou": 0.0,
                "ground_center_hit": False,
                "ground_extreme_box": False,
                "ground_num_detections": 0,
            }
        image_path = str(example.get("clean_rn_image_path") or "")
        if not image_path:
            image_path = str(row_image_path(self.package_root, example))
        width, height = image_size(image_path)
        gold = normalize_bbox(get_gold_bbox(example), width, height)
        result = predict_groundingdino(
            self.dino,
            image_path,
            ground_text,
            self.box_threshold,
            self.text_threshold,
            self.device,
        )
        top = result["predictions"][0] if result["predictions"] else None
        pred = top.get("box_xyxy") if isinstance(top, dict) else None
        iou = bbox_iou(pred, gold)
        return {
            "ground_detected": pred is not None,
            "ground_bbox": pred,
            "ground_score": top.get("score") if isinstance(top, dict) else None,
            "ground_iou": iou,
            "ground_center_hit": center_hit(pred, gold),
            "ground_extreme_box": is_extreme_box(pred, gold, width, height),
            "ground_num_detections": len(result["predictions"]),
        }

    def retrieval_metrics(self, example: dict[str, Any], query: str) -> dict[str, Any]:
        target_ids = set(self.targets.get(str(example.get("sample_id") or ""), []))
        if not self.bm25 or not target_ids or not query.strip():
            return {"retrieval_r1": None, "retrieval_r5": None, "retrieval_mrr": None, "retrieval_top_ids": []}
        ranked = self.bm25.rank(query, k=50)
        top_ids = [doc_id for doc_id, _score in ranked]
        return {
            "retrieval_r1": bool(set(top_ids[:1]) & target_ids),
            "retrieval_r5": bool(set(top_ids[:5]) & target_ids),
            "retrieval_mrr": reciprocal_rank(ranked, target_ids),
            "retrieval_top_ids": top_ids[:5],
        }

    def score(self, example: dict[str, Any], prediction: str, reward_mode: str = "dagig_grounded") -> dict[str, Any]:
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"Unknown reward_mode={reward_mode}")
        pred = extract_segments(prediction, TAGS)
        target = example.get("target_segments") if isinstance(example.get("target_segments"), dict) else {}
        answer_target = str(example.get("gold_answer") or example.get("answer") or target.get("answer") or "")
        teacher_ground = str(example.get("teacher_ground_expression") or example.get("ground_expression") or target.get("ground") or "")
        teacher_observe = str(example.get("teacher_observation") or target.get("observe") or "")
        teacher_query = str(example.get("teacher_search_query") or target.get("search") or "")
        target_evidence = str(example.get("teacher_evidence_text") or target.get("evidence") or example.get("evidence") or "")
        anchor = str(example.get("semantic_anchor") or example.get("visual_anchor") or teacher_ground)

        malformed = not all(str(pred.get(tag, "")).strip() for tag in TAGS)
        format_valid = not malformed
        ground_text = pred.get("ground", "")
        observe = pred.get("observe", "")
        query = pred.get("search", "")
        evidence = pred.get("evidence", "")
        answer = pred.get("answer", "")

        ground_metrics = self.score_grounding(example, ground_text)
        retrieval = self.retrieval_metrics(example, query)
        evidence_f1 = token_f1(evidence, target_evidence)
        answer_em = 1.0 if contains_match(answer, answer_target) else 0.0
        answer_f1 = token_f1(answer, answer_target)
        ground_forbidden = bool(FORBIDDEN_RE.search(ground_text))
        observe_forbidden = bool(FORBIDDEN_RE.search(observe))
        query_answer_leak = answer_leaks(answer_target, query)
        ground_generic = too_generic_ground(ground_text)
        anchor_hit = query_anchor_hit(query, anchor)
        observe_anchor_hit = query_anchor_hit(observe, anchor) or token_f1(observe, teacher_observe) >= 0.20
        observe_query_overlap = token_f1(observe, query)
        query_related = anchor_hit or observe_query_overlap >= 0.15 or token_f1(query, teacher_query) >= 0.25
        query_specificity = query_specificity_score(query)
        evidence_supported = bool(evidence_f1 >= 0.20 or retrieval.get("retrieval_r5") is True)
        unsupported_answer = bool(answer.strip() and answer_em and not evidence_supported)
        spurious_success = bool(answer_em and (not evidence_supported or not query_related))
        no_detection = ground_metrics["ground_detected"] is False
        extreme_box = ground_metrics["ground_extreme_box"] is True
        iou = ground_metrics["ground_iou"]
        center = ground_metrics["ground_center_hit"] is True
        detected = ground_metrics["ground_detected"] is True
        grounding_good = detected and (center or (iou is not None and iou >= 0.10)) and not ground_generic and not ground_forbidden

        R_ground = 0.0
        if iou is not None and iou >= 0.5:
            R_ground += 1.0
        elif iou is not None and iou >= 0.3:
            R_ground += 0.6
        if center:
            R_ground += 0.3
        if detected:
            R_ground += 0.1
        if no_detection:
            R_ground -= 0.2
        if extreme_box:
            R_ground -= 0.2
        if ground_generic:
            R_ground -= 0.2
        if ground_forbidden:
            R_ground -= 0.2

        R_observe = 0.0
        if grounding_good:
            if observe_anchor_hit:
                R_observe += 0.3
            if token_f1(observe, teacher_observe) >= 0.20:
                R_observe += 0.2
        if observe_forbidden:
            R_observe -= 0.3
        if observe.strip() and not observe_anchor_hit and token_f1(observe, teacher_observe) < 0.05:
            R_observe -= 0.3

        R_search = 0.0
        if query_related:
            if anchor_hit:
                R_search += 0.4
            if query_specificity >= 0.45:
                R_search += 0.3
            if retrieval.get("retrieval_r5") is True:
                R_search += 0.4
            if retrieval.get("retrieval_r1") is True:
                R_search += 0.2
        else:
            R_search -= 0.3 if query.strip() else 0.0
        if query_answer_leak:
            R_search -= 0.3

        R_evidence = 0.5 if evidence_supported else -0.5

        R_answer = 0.0
        if evidence_supported:
            if answer_em:
                R_answer += 1.0
            else:
                R_answer += 0.5 * answer_f1
        elif answer_em:
            R_answer -= 0.5
        if answer.strip() and evidence.strip() and token_f1(answer, evidence) == 0.0 and not evidence_supported:
            R_answer -= 0.5

        R_cost = 0.0
        if malformed:
            R_cost -= 0.1
        decision = normalize(pred.get("search_decision", ""))
        if decision == "no search" and query.strip():
            R_cost -= 0.05
        if decision == "search" and not query.strip():
            R_cost -= 0.05

        if reward_mode == "outcome_only":
            reward_total = (1.0 if answer_em else 0.5 * answer_f1) + (-0.2 if malformed else 0.0)
            mode_components = {"R_ground": 0.0, "R_observe": 0.0, "R_search": 0.0, "R_evidence": 0.0, "R_answer": reward_total, "R_cost": -0.2 if malformed else 0.0}
        elif reward_mode == "outcome_plus_ground_penalty":
            base = 1.0 if answer_em else 0.5 * answer_f1
            penalties = 0.0
            penalties -= 0.2 if malformed or not ground_text.strip() else 0.0
            penalties -= 0.2 if no_detection else 0.0
            penalties -= 0.2 if extreme_box else 0.0
            penalties -= 0.1 if not query.strip() else 0.0
            reward_total = base + penalties
            mode_components = {"R_ground": penalties, "R_observe": 0.0, "R_search": 0.0, "R_evidence": 0.0, "R_answer": base, "R_cost": penalties}
        elif reward_mode == "generic_process":
            Rg = 0.2 * float(bool(ground_text.strip())) + 0.2 * float(detected) + 0.2 * float(center)
            Ro = 0.2 * float(bool(observe.strip())) + 0.2 * token_f1(observe, teacher_observe)
            Rs = 0.2 * float(bool(query.strip())) + 0.2 * query_specificity + 0.2 * float(retrieval.get("retrieval_r5") is True)
            Re = 0.2 * float(evidence_supported)
            Ra = 0.2 * answer_em + 0.2 * answer_f1
            Rc = -0.2 if malformed else 0.0
            reward_total = Rg + Ro + Rs + Re + Ra + Rc
            mode_components = {"R_ground": Rg, "R_observe": Ro, "R_search": Rs, "R_evidence": Re, "R_answer": Ra, "R_cost": Rc}
        else:
            reward_total = R_ground + R_observe + R_search + R_evidence + R_answer + R_cost
            mode_components = {
                "R_ground": R_ground,
                "R_observe": R_observe,
                "R_search": R_search,
                "R_evidence": R_evidence,
                "R_answer": R_answer,
                "R_cost": R_cost,
            }

        return {
            "reward_mode": reward_mode,
            "reward_total": float(reward_total),
            **{k: float(v) for k, v in mode_components.items()},
            "malformed": bool(malformed),
            "format_valid": bool(format_valid),
            "unsupported_answer": bool(unsupported_answer),
            "spurious_success": bool(spurious_success),
            "ground_iou": iou,
            "ground_center_hit": center if ground_metrics["ground_center_hit"] is not None else None,
            "ground_detected": detected if ground_metrics["ground_detected"] is not None else None,
            "ground_score": ground_metrics["ground_score"],
            "ground_bbox": ground_metrics["ground_bbox"],
            "ground_extreme_box": ground_metrics["ground_extreme_box"],
            "ground_num_detections": ground_metrics["ground_num_detections"],
            "retrieval_r1": retrieval["retrieval_r1"],
            "retrieval_r5": retrieval["retrieval_r5"],
            "retrieval_mrr": retrieval["retrieval_mrr"],
            "retrieval_top_ids": retrieval["retrieval_top_ids"],
            "answer_em": float(answer_em),
            "answer_f1": float(answer_f1),
            "query_anchor_hit": bool(anchor_hit),
            "query_specificity": float(query_specificity),
            "evidence_support": bool(evidence_supported),
            "evidence_f1": float(evidence_f1),
            "ground_non_empty": bool(ground_text.strip()),
            "ground_forbidden_word": bool(ground_forbidden),
            "ground_too_generic": bool(ground_generic),
            "query_related_to_ground": bool(query_related),
            "query_answer_leakage": bool(query_answer_leak),
            "prediction_segments": pred,
        }


def load_rollout_predictions(path: str | Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row.get("rollouts"), list):
            for idx, rollout in enumerate(row["rollouts"]):
                text = rollout.get("completion") or rollout.get("prediction") or rollout.get("text") or ""
                out.append(
                    {
                        "sample_id": row.get("sample_id") or rollout.get("sample_id"),
                        "rollout_index": idx,
                        "prediction": text,
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
        "reward_total": avg("reward_total"),
        "malformed_rate": avg("malformed"),
        "ground_detection_rate": avg("ground_detected"),
        "mean_iou": avg("ground_iou"),
        "center_hit": avg("ground_center_hit"),
        "retrieval_r1": avg("retrieval_r1"),
        "retrieval_r5": avg("retrieval_r5"),
        "retrieval_mrr": avg("retrieval_mrr"),
        "answer_em": avg("answer_em"),
        "answer_f1": avg("answer_f1"),
        "R_ground": avg("R_ground"),
        "R_observe": avg("R_observe"),
        "R_search": avg("R_search"),
        "R_evidence": avg("R_evidence"),
        "R_answer": avg("R_answer"),
        "R_cost": avg("R_cost"),
    }


def main() -> int:
    args = parse_args()
    examples = load_jsonl(args.rl_data_jsonl, limit=args.limit)
    by_id = {str(row.get("sample_id")): row for row in examples}
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

    scorer = GroundedRewardScorer(
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
    scored_rows: list[dict[str, Any]] = []
    for row in rollout_rows:
        sample_id = str(row.get("sample_id") or "")
        example = by_id.get(sample_id)
        if example is None:
            continue
        score = scorer.score(example, str(row.get("prediction") or ""), args.reward_mode)
        scored_rows.append({"sample_id": sample_id, "rollout_index": row.get("rollout_index", 0), "prediction": row.get("prediction"), **score})
    write_jsonl(args.output_jsonl, scored_rows)
    summary = aggregate(scored_rows)
    summary.update({"reward_mode": args.reward_mode, "rl_data_jsonl": args.rl_data_jsonl, "rollouts_jsonl": args.rollouts_jsonl})
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
