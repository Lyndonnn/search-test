#!/usr/bin/env python3
"""Evaluate ground-action SFT generations and model-expression GroundingDINO."""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any

import torch

from grounded_pipeline_utils import (
    DINO_CONFIG,
    DINO_WEIGHTS,
    aggregate_grounding_metrics,
    bbox_iou,
    center_hit,
    evidence_text,
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
OLD_DIRECT_BBOX = {"mean_iou": 0.0358, "iou_ge_0_3": 0.0265}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_predictions_jsonl", required=True)
    parser.add_argument("--output_metrics_json", required=True)
    parser.add_argument("--summary_md", default="")
    parser.add_argument("--model_name_or_path", default="/data/zhengxiang/code/dagig/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter_dir", default="checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora")
    parser.add_argument("--package_root", default="data/dagig_rn03_10_ground_expr_v3_full")
    parser.add_argument("--retrieval_corpus", default="data/dagig_rn03_10_retrieval/corpus.jsonl")
    parser.add_argument("--box_threshold", type=float, default=0.10)
    parser.add_argument("--text_threshold", type=float, default=0.10)
    parser.add_argument("--grounding_config", default=DINO_CONFIG)
    parser.add_argument("--grounding_weights", default=DINO_WEIGHTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--require_beats_old_direct_bbox", action="store_true")
    return parser.parse_args()


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def contains_match(pred: str, target: str) -> bool:
    p = norm(pred)
    t = norm(target)
    return bool(p and t and (p in t or t in p))


def query_anchor_hit(query: str, anchor: str) -> bool:
    query_tokens = set(tokenize(query))
    anchor_tokens = [tok for tok in tokenize(anchor) if len(tok) > 1]
    return bool(anchor_tokens and any(tok in query_tokens for tok in anchor_tokens))


def query_specificity_score(query: str) -> float:
    toks = tokenize(query)
    if not toks:
        return 0.0
    length_score = min(len(toks) / 8.0, 1.0)
    numeric_or_named = any(tok.isdigit() or len(tok) >= 5 for tok in toks)
    return min(1.0, 0.7 * length_score + (0.3 if numeric_or_named else 0.0))


def answer_leaks(answer: Any, text: str) -> bool:
    answer_norm = norm(str(answer or ""))
    text_norm = norm(text)
    return bool(answer_norm and len(answer_norm) >= 2 and answer_norm in text_norm)


def forbidden_ground(text: str) -> bool:
    return bool(re.search(r"bbox|bounding box|red\s*box|red-box|annotation|annotated", str(text), flags=re.I))


def build_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = str(example["prompt"])
    images = [p for p in example.get("images", []) if p]
    content: list[dict[str, Any]] = []
    for image in images:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        min_pixels=256 * 28 * 28,
        max_pixels=args.max_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    return model, processor


@torch.inference_mode()
def generate_one(model: Any, processor: Any, example: dict[str, Any], max_new_tokens: int) -> str:
    from qwen_vl_utils import process_vision_info

    messages = build_messages(example)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def load_corpus(path: str) -> list[dict[str, str]]:
    corpus_path = Path(path)
    if not corpus_path.is_file():
        return []
    rows = []
    for idx, row in enumerate(load_jsonl(corpus_path)):
        text = str(row.get("text") or row.get("contents") or row.get("content") or row.get("document") or "")
        doc_id = str(row.get("doc_id") or row.get("id") or row.get("url") or idx)
        if text:
            rows.append({"id": doc_id, "text": text})
    return rows


def retrieve_rank(query: str, target_evidence: str, corpus: list[dict[str, str]], k: int = 50) -> int | None:
    if not query.strip() or not target_evidence.strip() or not corpus:
        return None
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return None
    target_norm = norm(target_evidence)
    scored = []
    for doc in corpus:
        doc_tokens = set(tokenize(doc["text"]))
        overlap = len(q_tokens & doc_tokens)
        if overlap <= 0:
            continue
        scored.append((overlap / max(1, len(q_tokens)), doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    for rank, (_, doc) in enumerate(scored[:k], start=1):
        doc_norm = norm(doc["text"])
        if target_norm and (target_norm in doc_norm or doc_norm in target_norm):
            return rank
    return None


def score_text_metrics(example: dict[str, Any], prediction: str, corpus: list[dict[str, str]]) -> dict[str, Any]:
    pred = extract_segments(prediction, TAGS)
    target = example.get("target_segments")
    if not isinstance(target, dict):
        target = extract_segments(str(example.get("target", "")), TAGS)
    target_evidence = str(target.get("evidence") or evidence_text(example))
    answer_target = str(target.get("answer") or example.get("answer") or "")
    query = pred.get("search", "")
    answer_pred = pred.get("answer", "")
    rank = retrieve_rank(query, target_evidence, corpus)
    anchor = str(example.get("visual_anchor") or example.get("semantic_anchor") or example.get("ground_expression") or "")
    answer_em = contains_match(answer_pred, answer_target)
    evidence_f1 = token_f1(pred.get("evidence", ""), target_evidence)
    anchor_hit = query_anchor_hit(query, anchor)
    return {
        "valid_format": all(bool(pred.get(tag, "").strip()) for tag in TAGS),
        "malformed": not all(bool(pred.get(tag, "").strip()) for tag in TAGS),
        "ground_non_empty": bool(pred.get("ground", "").strip()),
        "ground_teacher_token_f1": token_f1(pred.get("ground", ""), str(target.get("ground") or example.get("ground_expression") or "")),
        "ground_forbidden_word": forbidden_ground(pred.get("ground", "")),
        "ground_answer_leakage": answer_leaks(answer_target, pred.get("ground", "")),
        "query_anchor_hit": anchor_hit,
        "query_specificity": query_specificity_score(query),
        "retrieval_rank": rank,
        "r_at_1": rank == 1,
        "r_at_5": bool(rank and rank <= 5),
        "mrr": 1.0 / rank if rank else 0.0,
        "evidence_f1": evidence_f1,
        "answer_em": answer_em,
        "answer_f1": token_f1(answer_pred, answer_target),
        "unsupported": bool(answer_pred.strip()) and answer_em and evidence_f1 < 0.20,
        "spurious_success": bool(answer_em) and (not anchor_hit or evidence_f1 < 0.20),
        "segments": pred,
        "target_segments": target,
    }


def score_grounding(args: argparse.Namespace, examples: list[dict[str, Any]], scored_rows: list[dict[str, Any]]) -> None:
    dino = load_groundingdino_model(args.grounding_config, args.grounding_weights, args.device)
    for ex, row in zip(examples, scored_rows):
        ground = row["segments"].get("ground", "")
        image_path = row_image_path(args.package_root, ex)
        width, height = image_size(image_path)
        gold = normalize_bbox(get_gold_bbox(ex), width, height)
        pred_box = None
        score = None
        phrase = ""
        num = 0
        if ground.strip():
            result = predict_groundingdino(dino, image_path, ground, args.box_threshold, args.text_threshold, args.device)
            preds = result["predictions"]
            num = len(preds)
            if preds:
                top = preds[0]
                pred_box = top.get("box_xyxy")
                score = top.get("score")
                phrase = top.get("phrase", "")
        iou = bbox_iou(pred_box, gold)
        row.update(
            {
                "model_ground_image_path": str(image_path),
                "model_ground_gold_bbox_xyxy": gold,
                "model_ground_pred_bbox_xyxy": pred_box,
                "model_ground_score": score,
                "model_ground_phrase": phrase,
                "model_ground_num_detections": num,
                "model_ground_detected": pred_box is not None,
                "model_ground_iou": iou,
                "model_ground_iou_ge_0_1": iou >= 0.1,
                "model_ground_iou_ge_0_3": iou >= 0.3,
                "model_ground_iou_ge_0_5": iou >= 0.5,
                "model_ground_center_hit": center_hit(pred_box, gold),
                "model_ground_extreme_box": is_extreme_box(pred_box, gold, width, height),
            }
        )


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    def mean(key: str) -> float:
        return sum(float(row.get(key, 0.0) or 0.0) for row in rows) / n

    grounding_rows = [
        {
            "detected": row.get("model_ground_detected"),
            "iou": row.get("model_ground_iou"),
            "iou_ge_0_1": row.get("model_ground_iou_ge_0_1"),
            "iou_ge_0_3": row.get("model_ground_iou_ge_0_3"),
            "iou_ge_0_5": row.get("model_ground_iou_ge_0_5"),
            "center_hit": row.get("model_ground_center_hit"),
            "best_score": row.get("model_ground_score"),
            "pred_gold_area_ratio": None,
            "extreme_box": row.get("model_ground_extreme_box"),
        }
        for row in rows
    ]
    g = aggregate_grounding_metrics(grounding_rows)
    out = {
        "n": n,
        "valid_format_rate": mean("valid_format"),
        "malformed_rate": mean("malformed"),
        "ground_non_empty_rate": mean("ground_non_empty"),
        "teacher_expression_token_f1": mean("ground_teacher_token_f1"),
        "forbidden_word_rate": mean("ground_forbidden_word"),
        "final_answer_leakage_rate": mean("ground_answer_leakage"),
        "model_expression_dino_detection_rate": g.get("detection_rate", 0.0),
        "model_expression_dino_no_detection_rate": g.get("no_detection_rate", 0.0),
        "model_expression_dino_mean_iou": g.get("mean_iou", 0.0),
        "model_expression_dino_median_iou": g.get("median_iou", 0.0),
        "model_expression_dino_iou_ge_0_1": g.get("iou_ge_0_1", 0.0),
        "model_expression_dino_iou_ge_0_3": g.get("iou_ge_0_3", 0.0),
        "model_expression_dino_iou_ge_0_5": g.get("iou_ge_0_5", 0.0),
        "model_expression_dino_center_hit": g.get("center_hit_rate", 0.0),
        "r_at_1": mean("r_at_1"),
        "r_at_5": mean("r_at_5"),
        "mrr": mean("mrr"),
        "query_anchor_hit": mean("query_anchor_hit"),
        "query_specificity": mean("query_specificity"),
        "unsupported_rate": mean("unsupported"),
        "spurious_success_rate": mean("spurious_success"),
        "answer_em": mean("answer_em"),
        "answer_f1": mean("answer_f1"),
        "old_direct_bbox_mean_iou": OLD_DIRECT_BBOX["mean_iou"],
        "old_direct_bbox_iou_ge_0_3": OLD_DIRECT_BBOX["iou_ge_0_3"],
    }
    out["beats_old_direct_bbox"] = bool(
        out["model_expression_dino_mean_iou"] > OLD_DIRECT_BBOX["mean_iou"]
        and out["model_expression_dino_iou_ge_0_3"] > OLD_DIRECT_BBOX["iou_ge_0_3"]
    )
    return out


def write_summary(path: str | Path, metrics: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ground-Action SFT Evaluation",
        "",
        f"- n: {metrics.get('n')}",
        f"- valid format: {metrics.get('valid_format_rate'):.4f}",
        f"- model expression + DINO mean IoU: {metrics.get('model_expression_dino_mean_iou'):.4f}",
        f"- model expression + DINO IoU>=0.3: {metrics.get('model_expression_dino_iou_ge_0_3'):.4f}",
        f"- model expression + DINO center-hit: {metrics.get('model_expression_dino_center_hit'):.4f}",
        f"- answer EM/F1: {metrics.get('answer_em'):.4f} / {metrics.get('answer_f1'):.4f}",
        f"- beats old direct bbox: {metrics.get('beats_old_direct_bbox')}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    examples = load_jsonl(args.eval_file, limit=args.limit)
    if not examples:
        raise ValueError(f"No eval examples loaded from {args.eval_file}")
    corpus = load_corpus(args.retrieval_corpus)

    model, processor = load_model(args)
    rows: list[dict[str, Any]] = []
    for ex in examples:
        prediction = generate_one(model, processor, ex, args.max_new_tokens)
        scored = score_text_metrics(ex, prediction, corpus)
        rows.append(
            {
                "sample_id": ex.get("sample_id"),
                "split": ex.get("split"),
                "question": ex.get("question"),
                "answer": ex.get("answer"),
                "prediction": prediction,
                **{k: v for k, v in scored.items() if k not in {"segments", "target_segments"}},
                "segments": scored["segments"],
                "target_segments": scored["target_segments"],
            }
        )

    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    score_grounding(args, examples, rows)
    metrics = aggregate(rows)
    metrics.update(
        {
            "eval_file": args.eval_file,
            "adapter_dir": args.adapter_dir,
            "model_name_or_path": args.model_name_or_path,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "retrieval_corpus": args.retrieval_corpus,
        }
    )
    write_jsonl(args.output_predictions_jsonl, rows)
    write_json(args.output_metrics_json, metrics)
    if args.summary_md:
        write_summary(args.summary_md, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.require_beats_old_direct_bbox and not metrics.get("beats_old_direct_bbox"):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
