#!/usr/bin/env python3
"""Re-score the evidence->answer edge with supported-answer verifier v2."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import mean
from typing import Any

from grounded_pipeline_utils import extract_segments, load_jsonl, md_table, write_jsonl


VERIFIER_PATH = Path(__file__).with_name("49_supported_answer_verifier_v2.py")
SPEC = importlib.util.spec_from_file_location("supported_answer_verifier_v2", VERIFIER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import verifier from {VERIFIER_PATH}")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
verify_supported_answer = VERIFIER.verify_supported_answer
answer_appears_in_text = VERIFIER.answer_appears_in_text


SPLITS = ["train", "dev", "test"]
TAGS = ["ground", "observe", "search_decision", "search", "evidence", "answer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored_dir", default="results/dagig_rn03_10_counterfactual_dagig_v3")
    parser.add_argument("--rl_data_dir", default="data/dagig_rn03_10_grounded_rl")
    parser.add_argument("--counterfactual_dir", default="data/dagig_rn03_10_counterfactuals")
    parser.add_argument("--verifier_dir", default="results/dagig_rn03_10_counterfactual_dagig_v31")
    parser.add_argument("--out_dir", default="results/dagig_rn03_10_counterfactual_dagig_v31")
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    return parser.parse_args()


def load_map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in load_jsonl(path)}


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return False


def real_score(verifier: dict[str, Any]) -> float:
    return (
        0.40 * float(verifier.get("evidence_supports_prediction") is True)
        + 0.30 * float(verifier.get("supported_answer_v2") is True)
        + 0.20 * float(verifier.get("answer_f1") or 0.0)
        + 0.10 * float(verifier.get("answer_type_valid") is True)
    )


def answer_leak_before_answer(segments: dict[str, str], predicted_answer: str, question: str) -> bool:
    if not str(predicted_answer or "").strip():
        return False
    pre_answer = "\n".join(str(segments.get(tag, "")) for tag in ["ground", "observe", "search_decision", "search"])
    return answer_appears_in_text(predicted_answer, pre_answer, question)


def cf_evidence_items(cf: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    counterfactuals = cf.get("counterfactuals")
    if isinstance(counterfactuals, dict):
        items.extend(counterfactuals.get("evidence_cf") or [])
    items.append({"type": "no_evidence", "text": ""})
    return items


def score_counterfactuals(
    example: dict[str, Any],
    predicted_answer: str,
    semantic_anchor: str,
    cf: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    scored = []
    for item in cf_evidence_items(cf):
        evidence = str(item.get("text") or item.get("evidence") or "")
        verifier = verify_supported_answer(
            question=example.get("question"),
            gold_answer=example.get("gold_answer") or example.get("answer"),
            predicted_answer=predicted_answer,
            retrieved_evidence=example.get("teacher_evidence_text") or example.get("evidence"),
            selected_evidence=evidence,
            semantic_anchor=semantic_anchor,
            question_type=example.get("question_type"),
        )
        score = real_score(verifier)
        scored.append({"type": item.get("type", "counterfactual_evidence"), "score": score, "verifier": verifier})
    best = max((float(item["score"]) for item in scored), default=0.0)
    return scored, best


def score_split(split: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    scored_rows = load_jsonl(Path(args.scored_dir) / f"scored_rollouts_{split}.jsonl")
    examples = load_map(Path(args.rl_data_dir) / f"grounded_rl_{split}.jsonl")
    cfs = load_map(Path(args.counterfactual_dir) / f"counterfactual_{split}.jsonl")
    verifier_rows = {
        (str(row.get("sample_id")), int(row.get("rollout_index") or 0)): row
        for row in load_jsonl(Path(args.verifier_dir) / f"supported_answer_v2_{split}.jsonl")
    }
    out = []
    for row in scored_rows:
        sample_id = str(row.get("sample_id") or "")
        rollout_index = int(row.get("rollout_index") or 0)
        example = examples.get(sample_id, {})
        cf = cfs.get(sample_id, {})
        segments = row.get("prediction_segments")
        if not isinstance(segments, dict):
            segments = extract_segments(str(row.get("prediction") or ""), TAGS)
        verifier = verifier_rows.get((sample_id, rollout_index))
        if verifier is None:
            verifier = verify_supported_answer(
                question=example.get("question"),
                gold_answer=example.get("gold_answer") or example.get("answer"),
                predicted_answer=segments.get("answer", ""),
                retrieved_evidence=example.get("teacher_evidence_text") or example.get("evidence"),
                selected_evidence=segments.get("evidence", ""),
                semantic_anchor=example.get("semantic_anchor") or example.get("visual_anchor"),
                question_type=example.get("question_type"),
            )
        semantic_anchor = str(example.get("semantic_anchor") or example.get("visual_anchor") or "")
        cf_scores, best_cf = score_counterfactuals(example, str(segments.get("answer", "")), semantic_anchor, cf)
        r_score = real_score(verifier)
        raw_credit = r_score - best_cf
        leak = answer_leak_before_answer(segments, str(segments.get("answer", "")), str(example.get("question") or ""))
        evidence_supports_prediction = bool(verifier.get("evidence_supports_prediction"))
        answer_correct = bool(verifier.get("answer_correct"))
        unsupported_answer = bool(answer_correct and not evidence_supports_prediction)
        spurious_answer = bool(answer_correct and not evidence_supports_prediction)
        gate_pass = bool(not leak and evidence_supports_prediction and answer_correct)
        r_answer = raw_credit if gate_pass else min(raw_credit, 0.0)
        r_cost_v31 = max(0.0, -float(row.get("R_cost") or 0.0))
        if leak:
            r_cost_v31 += 0.20
        if unsupported_answer or spurious_answer:
            r_cost_v31 += 0.15
        edge = {
            "real_score": float(r_score),
            "cf_best_score": float(best_cf),
            "delta_best": float(raw_credit),
            "edge_credit": float(r_answer),
            "gate_pass": bool(gate_pass),
        }
        edges = dict(row.get("edges") or {})
        edges["evidence_to_answer_v31"] = edge
        out.append(
            {
                **row,
                "split": split,
                "prediction_segments": segments,
                "R_answer_old": row.get("R_answer"),
                "R_answer_v31": float(r_answer),
                "R_cost_v31": float(r_cost_v31),
                "evidence_answer_real_score_v31": float(r_score),
                "evidence_answer_best_cf_score_v31": float(best_cf),
                "evidence_answer_raw_delta_v31": float(raw_credit),
                "evidence_answer_gate_pass_v31": bool(gate_pass),
                "answer_leak_before_answer_v31": bool(leak),
                "unsupported_answer_v31": bool(unsupported_answer),
                "spurious_answer_v31": bool(spurious_answer),
                "answer_correct_v2": bool(verifier.get("answer_correct")),
                "answer_type_valid_v2": bool(verifier.get("answer_type_valid")),
                "answer_f1_v2": float(verifier.get("answer_f1") or 0.0),
                "answer_em_v2": float(verifier.get("answer_em") or 0.0),
                "evidence_supports_gold_v2": bool(verifier.get("evidence_supports_gold")),
                "evidence_supports_prediction_v2": bool(verifier.get("evidence_supports_prediction")),
                "prediction_supported_by_evidence_v2": bool(verifier.get("prediction_supported_by_evidence")),
                "supported_answer_v2": bool(verifier.get("supported_answer_v2")),
                "support_score_v2": float(verifier.get("support_score_v2") or 0.0),
                "supported_answer_failure_type_v2": verifier.get("failure_type"),
                "supported_answer_verifier_v2": verifier,
                "evidence_answer_cf_scores_v31": cf_scores,
                "edges": edges,
            }
        )
    write_jsonl(Path(args.out_dir) / f"scored_rollouts_{split}.jsonl", out)
    return out


def summarize(rows_by_split: dict[str, list[dict[str, Any]]]) -> str:
    summary = []
    for split, rows in rows_by_split.items():
        n = len(rows)
        summary.append(
            {
                "split": split,
                "n": n,
                "R_answer_old": mean([float(r.get("R_answer_old") or 0.0) for r in rows]) if n else 0.0,
                "R_answer_v31": mean([float(r.get("R_answer_v31") or 0.0) for r in rows]) if n else 0.0,
                "supported_answer_v2": mean([float(r.get("supported_answer_v2") is True) for r in rows]) if n else 0.0,
                "answer_correct_v2": mean([float(r.get("answer_correct_v2") is True) for r in rows]) if n else 0.0,
                "evidence_supports_prediction_v2": mean([float(r.get("evidence_supports_prediction_v2") is True) for r in rows]) if n else 0.0,
                "gate_pass": mean([float(r.get("evidence_answer_gate_pass_v31") is True) for r in rows]) if n else 0.0,
            }
        )
    return "# Evidence->Answer Edge v3.1 Summary\n\n" + md_table(summary) + "\n"


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = {split: score_split(split, args) for split in args.splits}
    (out_dir / "evidence_answer_edge_v31_summary.md").write_text(summarize(rows_by_split), encoding="utf-8")
    print(json.dumps({split: len(rows) for split, rows in rows_by_split.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
