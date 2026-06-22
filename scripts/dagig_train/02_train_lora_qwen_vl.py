#!/usr/bin/env python3
"""Small weighted LoRA/QLoRA SFT for DAG-IG Pix2Fact ablations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=1)
    return parser.parse_args()


def load_jsonl(path: str, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if float(row.get("loss_weight", 0.0) or 0.0) <= 0:
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def build_messages(example: dict[str, Any], include_answer: bool) -> list[dict[str, Any]]:
    prompt = str(example["prompt"])
    target = str(example["target"])
    images = [p for p in example.get("images", []) if p]
    if images:
        content: list[dict[str, Any]] = []
        for image in images:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
    else:
        content = [{"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    if include_answer:
        messages.append({"role": "assistant", "content": target})
    return messages


class WeightedQwenVLCollator:
    def __init__(self, processor: Any, max_length: int):
        self.processor = processor
        self.max_length = max_length
        self.tokenizer = getattr(processor, "tokenizer", processor)

    @staticmethod
    def find_subsequence(haystack: list[int], needle: list[int], start_at: int = 0) -> int:
        if not needle or len(needle) > len(haystack):
            return -1
        first = needle[0]
        max_start = len(haystack) - len(needle)
        for start in range(max(start_at, 0), max_start + 1):
            if haystack[start] != first:
                continue
            if haystack[start : start + len(needle)] == needle:
                return start
        return -1

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def segment_texts(self, ex: dict[str, Any]) -> dict[str, str]:
        segments = ex.get("target_segments")
        if isinstance(segments, dict):
            return {str(k): str(v) for k, v in segments.items()}
        target = str(ex.get("target", ""))
        return {"target": target}

    def build_loss_weights(self, ex: dict[str, Any], input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weights = torch.zeros_like(labels, dtype=torch.float32)
        ids = input_ids[0].tolist()
        segment_weights = ex.get("segment_weights")
        if not isinstance(segment_weights, dict):
            segment_weights = {"target": float(ex.get("loss_weight", 1.0))}

        any_found = False
        for name, text in self.segment_texts(ex).items():
            block = f"<{name}>\n{text}\n</{name}>" if name != "target" else text
            encoded = self.encode_text(block)
            start = self.find_subsequence(ids, encoded)
            if start < 0 and name != "target":
                encoded = self.encode_text(str(text))
                start = self.find_subsequence(ids, encoded)
            if start >= 0:
                weight = float(segment_weights.get(name, ex.get("loss_weight", 1.0)) or 0.0)
                weights[:, start : start + len(encoded)] = weight
                any_found = True

        if not any_found:
            fallback = float(ex.get("loss_weight", 1.0) or 1.0)
            weights[labels != -100] = fallback
        else:
            positive = labels != -100
            unweighted_target = positive & (weights <= 0)
            if bool(unweighted_target.any()):
                min_positive = min(
                    [float(v) for v in segment_weights.values() if float(v) > 0.0] or [float(ex.get("loss_weight", 1.0) or 1.0)]
                )
                weights[unweighted_target] = min(min_positive, 0.05)
        weights[labels == -100] = 0.0
        return weights

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(features) != 1:
            raise ValueError(
                "This safety-first collator supports batch_size=1. "
                "Use --batch_size 1 and increase --grad_accum for this small ablation."
            )
        ex = features[0]
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as exc:  # pragma: no cover - exercised on server
            raise ImportError("qwen-vl-utils is required for visual SFT examples.") from exc

        full_messages = build_messages(ex, include_answer=True)
        prompt_messages = build_messages(ex, include_answer=False)
        full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(full_messages)

        model_inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = model_inputs["input_ids"].clone()
        prompt_len = min(prompt_inputs["input_ids"].shape[-1], labels.shape[-1])
        labels[:, :prompt_len] = -100
        labels[model_inputs["attention_mask"] == 0] = -100
        model_inputs["labels"] = labels
        model_inputs["loss_weights"] = self.build_loss_weights(ex, model_inputs["input_ids"], labels)
        return model_inputs


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        min_pixels=256 * 28 * 28,
        max_pixels=args.max_pixels,
    )
    quantization_config = None
    if args.use_qlora:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # pragma: no cover
            raise ImportError("bitsandbytes/transformers BitsAndBytesConfig is required for --use_qlora") from exc
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model, processor


def add_lora(model: Any, args: argparse.Namespace) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Use --batch_size 1 for the first reproducible DAG-IG SFT ablation.")
    rows = load_jsonl(args.train_file, limit=args.limit)
    if args.smoke_test:
        rows = rows[: min(2, len(rows))]
    if not rows:
        raise ValueError(f"No positive-weight training rows loaded from {args.train_file}")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args) | {"n_rows": len(rows)}, f, ensure_ascii=False, indent=2)

    model, processor = load_model_and_processor(args)
    model = add_lora(model, args)

    from transformers import Trainer, TrainingArguments

    class WeightedTrainer(Trainer):
        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any):
            loss_weights = inputs.pop("loss_weights").to(model.device)
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if labels is None:
                loss = outputs.loss
            else:
                labels = labels.to(logits.device)
                shift_logits = logits[..., :-1, :].contiguous().float()
                shift_labels = labels[..., 1:].contiguous()
                shift_weights = loss_weights[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
                token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                token_loss = token_loss.view_as(shift_labels)
                valid = shift_labels.ne(-100).float()
                weighted_valid = shift_weights * valid
                denom = weighted_valid.sum().clamp_min(1.0)
                loss = (token_loss * weighted_valid).sum() / denom
            return (loss, outputs) if return_outputs else loss

    save_strategy = "steps" if args.save_steps > 0 else "epoch"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=SFTDataset(rows),
        data_collator=WeightedQwenVLCollator(processor, args.max_length),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"saved LoRA adapter and processor to {args.output_dir}")


if __name__ == "__main__":
    main()
