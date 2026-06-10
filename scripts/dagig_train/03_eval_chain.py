#!/usr/bin/env python3
"""Evaluate base or LoRA adapter on DAG-IG SFT chain tasks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter_dir", default="")
    parser.add_argument("--max_new_tokens", type=int, default=64)
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


def token_f1(pred: str, target: str) -> float:
    pred_tokens = normalize(pred).split()
    target_tokens = normalize(target).split()
    if not pred_tokens or not target_tokens:
        return 0.0
    pred_counts = defaultdict(int)
    target_counts = defaultdict(int)
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


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["task_type"]].append(row)
        groups["all"].append(row)
    out = []
    for task_type, items in groups.items():
        out.append(
            {
                "task_type": task_type,
                "n": len(items),
                "mean_token_f1": sum(float(r["token_f1"]) for r in items) / len(items),
                "substring_match_rate": sum(1.0 if r["substring_match"] else 0.0 for r in items) / len(items),
                "valid_nonempty_rate": sum(1.0 if r["prediction"].strip() else 0.0 for r in items) / len(items),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    examples = load_jsonl(args.eval_file, args.limit)
    if not examples:
        raise ValueError(f"No eval examples in {args.eval_file}")
    model, processor = load_model(args)
    rows = []
    for ex in examples:
        pred = generate_one(model, processor, ex, args.max_new_tokens)
        target = str(ex.get("target", ""))
        rows.append(
            {
                "sample_id": ex.get("sample_id"),
                "variant": ex.get("variant"),
                "task_type": ex.get("task_type"),
                "prediction": pred,
                "target": target,
                "token_f1": token_f1(pred, target),
                "substring_match": contains_match(pred, target),
                "loss_weight": ex.get("loss_weight"),
            }
        )
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path = out_path.with_name(out_path.stem + "_details.csv")
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = aggregate(rows)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {out_path}")
    print(f"wrote {detail_path}")


if __name__ == "__main__":
    main()
