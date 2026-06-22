#!/usr/bin/env python3
"""Train a rejection-SFT LoRA continuation on DAG-IG preference pair winners."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import torch


SFT_PATH = Path(__file__).with_name("02_train_lora_qwen_vl.py")
SPEC = importlib.util.spec_from_file_location("dagig_sft_train", SFT_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import SFT utilities from {SFT_PATH}")
SFT_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SFT_MODULE)
SFTDataset = SFT_MODULE.SFTDataset
WeightedQwenVLCollator = SFT_MODULE.WeightedQwenVLCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_pairs", default="data/dagig_rn03_10_counterfactual_dagig/dagig_preference_pairs_train.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/dagig_rn03_10_counterfactual_dagig/dagig_dpo_7b_lora")
    parser.add_argument("--model_name_or_path", default="/data/zhengxiang/code/dagig/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--init_adapter_dir", default="checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    parser.add_argument("--max_length", type=int, default=3072)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=1)
    return parser.parse_args()


def load_pairs(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            chosen = str(row.get("chosen") or row.get("target") or "").strip()
            if not chosen:
                continue
            row["target"] = chosen
            row["loss_weight"] = float(row.get("loss_weight", 1.0) or 1.0)
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def load_model_processor(args: argparse.Namespace) -> tuple[Any, Any]:
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
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = PeftModel.from_pretrained(model, args.init_adapter_dir, is_trainable=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model, processor


def main() -> int:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Use --batch_size 1 and larger --grad_accum for this small rejection-SFT run.")
    rows = load_pairs(args.train_pairs, args.limit)
    if not rows:
        raise ValueError(f"No preference winners loaded from {args.train_pairs}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args) | {"n_rows": len(rows), "method": "rejection_sft"}, f, ensure_ascii=False, indent=2)

    model, processor = load_model_processor(args)
    from transformers import Trainer, TrainingArguments

    class WeightedTrainer(Trainer):
        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any):
            loss_weights = inputs.pop("loss_weights").to(model.device)
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weights[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_loss = token_loss.view_as(shift_labels)
            valid = shift_labels.ne(-100).float()
            denom = (shift_weights * valid).sum().clamp_min(1.0)
            loss = (token_loss * shift_weights * valid).sum() / denom
            return (loss, outputs) if return_outputs else loss

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="steps" if args.max_steps > 0 else "epoch",
        save_steps=max(10, args.max_steps // 2) if args.max_steps > 0 else 500,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
    )
    trainer = WeightedTrainer(
        model=model,
        args=train_args,
        train_dataset=SFTDataset(rows),
        data_collator=WeightedQwenVLCollator(processor, args.max_length),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"saved rejection-SFT LoRA adapter to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
