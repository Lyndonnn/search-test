#!/usr/bin/env python3
"""Sample multiple grounded SFT rollouts for counterfactual DAG-IG scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from grounded_pipeline_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/dagig_rn03_10_grounded_rl")
    parser.add_argument("--out_dir", default="results/dagig_rn03_10_counterfactual_dagig/rollouts")
    parser.add_argument("--model_name_or_path", default="/data/zhengxiang/code/dagig/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter_dir", default="checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    return parser.parse_args()


def shard_rows(rows: list[dict[str, Any]], shard_index: int, shard_count: int) -> list[dict[str, Any]]:
    if shard_count <= 1:
        return rows
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_index}")
    return [row for idx, row in enumerate(rows) if idx % shard_count == shard_index]


def split_output_path(out_dir: Path, split: str, shard_index: int, shard_count: int) -> Path:
    if shard_count <= 1:
        return out_dir / f"{split}.jsonl"
    return out_dir / f"{split}.shard{shard_index:02d}-of-{shard_count:02d}.jsonl"


def model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def build_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = str(example["prompt"])
    images = [p for p in example.get("images", []) if p]
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from peft import PeftModel
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
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    return model, processor


@torch.inference_mode()
def generate_k(model: Any, processor: Any, example: dict[str, Any], args: argparse.Namespace) -> list[str]:
    from qwen_vl_utils import process_vision_info

    messages = build_messages(example)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    device = model_device(model)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    generated = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        num_return_sequences=args.k,
        pad_token_id=getattr(processor.tokenizer, "eos_token_id", None),
    )
    new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_model(args)
    for split in args.splits:
        all_rows = load_jsonl(Path(args.data_dir) / f"grounded_rl_{split}.jsonl", limit=args.limit)
        rows = shard_rows(all_rows, args.shard_index, args.shard_count)
        out_path = split_output_path(out_dir, split, args.shard_index, args.shard_count)
        total = len(rows)
        with out_path.open("w", encoding="utf-8") as f:
            for idx, row in enumerate(rows, start=1):
                completions = generate_k(model, processor, row, args)
                for ridx, text in enumerate(completions):
                    f.write(
                        json.dumps(
                            {
                                "sample_id": row.get("sample_id"),
                                "split": split,
                                "rollout_index": ridx,
                                "prediction": text.strip(),
                                "source_adapter": args.adapter_dir,
                                "temperature": args.temperature,
                                "top_p": args.top_p,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                f.flush()
                if idx == 1 or idx % 10 == 0 or idx == total:
                    print(f"[rollout] {split} {idx}/{total} wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
