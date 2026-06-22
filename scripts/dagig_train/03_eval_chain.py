#!/usr/bin/env python3
"""Evaluate base or LoRA adapters on oracle-crop Pix2Fact-DAGIG chain tasks."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


SEGMENTS = ["observe", "search_decision", "search", "evidence", "answer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter_dir", default="")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path: str, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def token_f1(pred: str, target: str) -> float:
    pred_tokens = tokens(pred)
    target_tokens = tokens(target)
    if not pred_tokens or not target_tokens:
        return 0.0
    pred_counts: dict[str, int] = defaultdict(int)
    target_counts: dict[str, int] = defaultdict(int)
    for tok in pred_tokens:
        pred_counts[tok] += 1
    for tok in target_tokens:
        target_counts[tok] += 1
    common = sum(min(pred_counts[tok], target_counts[tok]) for tok in pred_counts)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_match(pred: str, target: str) -> bool:
    p = normalize(pred)
    t = normalize(target)
    return bool(p and t and (p in t or t in p))


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_segments(text: str) -> dict[str, str]:
    return {segment: extract_tag(text, segment) for segment in SEGMENTS}


def query_anchor_hit(query: str, anchor: str) -> bool:
    q_tokens = set(tokens(query))
    anchor_tokens = [tok for tok in tokens(anchor) if len(tok) > 1]
    if not anchor_tokens:
        return False
    return any(tok in q_tokens for tok in anchor_tokens)


def query_specificity_score(query: str) -> float:
    q = tokens(query)
    if not q:
        return 0.0
    length_score = min(len(q) / 8.0, 1.0)
    numeric_or_named = any(tok.isdigit() or len(tok) >= 5 for tok in q)
    return min(1.0, 0.7 * length_score + (0.3 if numeric_or_named else 0.0))


def build_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = str(example["prompt"])
    images = [p for p in example.get("images", []) if p]
    if images:
        content: list[dict[str, Any]] = []
        for image in images:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
    else:
        content = [{"type": "text", "text": prompt}]
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


def score_example(ex: dict[str, Any], prediction: str) -> dict[str, Any]:
    pred = extract_segments(prediction)
    target = ex.get("target_segments")
    if not isinstance(target, dict):
        target = extract_segments(str(ex.get("target", "")))
    answer_pred = pred.get("answer", "")
    answer_target = str(target.get("answer") or ex.get("answer") or "")
    query_pred = pred.get("search", "")
    evidence_pred = pred.get("evidence", "")
    anchor = str(ex.get("visual_anchor", ""))
    answer_em = contains_match(answer_pred, answer_target)
    answer_f1 = token_f1(answer_pred, answer_target)
    evidence_f1 = token_f1(evidence_pred, str(target.get("evidence", "")))
    anchor_hit = query_anchor_hit(query_pred, anchor)
    unsupported = bool(answer_pred.strip()) and answer_em and evidence_f1 < 0.20
    spurious = bool(answer_em) and (not anchor_hit or evidence_f1 < 0.20)
    return {
        "sample_id": ex.get("sample_id"),
        "variant": ex.get("variant"),
        "split": ex.get("split"),
        "prediction": prediction,
        "observe_prediction": pred.get("observe", ""),
        "search_decision_prediction": pred.get("search_decision", ""),
        "search_prediction": query_pred,
        "evidence_prediction": evidence_pred,
        "answer_prediction": answer_pred,
        "observe_target": str(target.get("observe", "")),
        "search_target": str(target.get("search", "")),
        "evidence_target": str(target.get("evidence", "")),
        "answer_target": answer_target,
        "observe_f1": token_f1(pred.get("observe", ""), str(target.get("observe", ""))),
        "query_f1": token_f1(query_pred, str(target.get("search", ""))),
        "evidence_f1": evidence_f1,
        "answer_f1": answer_f1,
        "answer_em": answer_em,
        "query_anchor_hit": anchor_hit,
        "query_specificity": query_specificity_score(query_pred),
        "valid_format": all(bool(pred.get(seg, "").strip()) for seg in ["observe", "search", "answer"]),
        "unsupported_answer": unsupported,
        "spurious_success": spurious,
        "visual_anchor": anchor,
        "question": ex.get("question"),
        "gold_answer": ex.get("answer"),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("variant", "unknown"))].append(row)
        groups["all"].append(row)
    out = []
    metric_keys = [
        "observe_f1",
        "query_f1",
        "evidence_f1",
        "answer_f1",
        "answer_em",
        "query_anchor_hit",
        "query_specificity",
        "valid_format",
        "unsupported_answer",
        "spurious_success",
    ]
    for variant, items in groups.items():
        row = {"variant": variant, "n": len(items)}
        for key in metric_keys:
            row[key] = sum(float(item[key]) for item in items) / max(1, len(items))
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    examples = load_jsonl(args.eval_file, args.limit)
    if not examples:
        raise ValueError(f"No eval examples in {args.eval_file}")
    model, processor = load_model(args)
    rows = [score_example(ex, generate_one(model, processor, ex, args.max_new_tokens)) for ex in examples]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path = out_path.with_name(out_path.stem + "_details.csv")
    detail_jsonl = out_path.with_name(out_path.stem + "_details.jsonl")
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with detail_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = aggregate(rows)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {out_path}")
    print(f"wrote {detail_path}")
    print(f"wrote {detail_jsonl}")


if __name__ == "__main__":
    main()
