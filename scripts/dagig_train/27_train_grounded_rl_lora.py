#!/usr/bin/env python3
"""Low-budget grounded RL LoRA continuation from the ground-action SFT adapter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
from collections import deque
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from grounded_pipeline_utils import load_jsonl, write_json


SCORER_PATH = Path(__file__).with_name("26_score_grounded_rollouts.py")
SPEC = importlib.util.spec_from_file_location("grounded_rollout_scorer", SCORER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import scorer from {SCORER_PATH}")
SCORER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER_MODULE)
GroundedRewardScorer = SCORER_MODULE.GroundedRewardScorer
REWARD_MODES = SCORER_MODULE.REWARD_MODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", default="data/dagig_rn03_10_grounded_rl/grounded_rl_train.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="/data/zhengxiang/code/dagig/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--init_adapter_dir", default="checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora")
    parser.add_argument("--reward_mode", choices=REWARD_MODES, required=True)
    parser.add_argument("--corpus_jsonl", default="data/dagig_rn03_10_retrieval/corpus.jsonl")
    parser.add_argument("--targets_json", default="data/dagig_rn03_10_retrieval/targets.json")
    parser.add_argument("--package_root", default="data/dagig_rn03_10_ground_expr_v3_full")
    parser.add_argument("--grounding_config", default=SCORER_MODULE.DINO_CONFIG)
    parser.add_argument("--grounding_weights", default=SCORER_MODULE.DINO_WEIGHTS)
    parser.add_argument("--box_threshold", type=float, default=0.10)
    parser.add_argument("--text_threshold", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rollout_n", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    parser.add_argument("--max_length", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--max_malformed_rate", type=float, default=0.30)
    return parser.parse_args()


def model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def build_messages(example: dict[str, Any], assistant_content: str | None = None) -> list[dict[str, Any]]:
    prompt = str(example["prompt"])
    images = [p for p in example.get("images", []) if p]
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})
    return messages


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
    model.train()
    return model, processor


@torch.no_grad()
def generate_rollouts(model: Any, processor: Any, example: dict[str, Any], args: argparse.Namespace) -> list[str]:
    from qwen_vl_utils import process_vision_info

    was_training = model.training
    model.eval()
    messages = build_messages(example)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    device = model_device(model)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.rollout_n,
            pad_token_id=getattr(processor.tokenizer, "eos_token_id", None),
        )
    finally:
        if was_training:
            model.train()
    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = outputs[:, prompt_len:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def sequence_logprob(model: Any, processor: Any, example: dict[str, Any], completion: str, max_length: int) -> torch.Tensor:
    from qwen_vl_utils import process_vision_info

    full_messages = build_messages(example, assistant_content=completion)
    prompt_messages = build_messages(example)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(full_messages)
    model_inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    device = model_device(model)
    model_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in model_inputs.items()}
    labels = model_inputs["input_ids"].clone()
    prompt_len = min(prompt_inputs["input_ids"].shape[-1], labels.shape[-1])
    labels[:, :prompt_len] = -100
    labels[model_inputs["attention_mask"] == 0] = -100
    outputs = model(**model_inputs)
    logits = outputs.logits[..., :-1, :].float()
    shift_labels = labels[..., 1:]
    valid = shift_labels.ne(-100)
    log_probs = F.log_softmax(logits, dim=-1)
    safe_labels = shift_labels.masked_fill(~valid, 0)
    token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logp * valid.float()).sum() / valid.float().sum().clamp_min(1.0)


def prepare_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [row for row in load_jsonl(args.train_file) if not row.get("leakage_sensitive_exclude")]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No usable RL rows loaded from {args.train_file}")
    return rows


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = prepare_rows(args)
    write_json(out_dir / "train_args.json", vars(args) | {"n_rows": len(rows)})

    scorer = GroundedRewardScorer(
        corpus_jsonl=args.corpus_jsonl,
        targets_json=args.targets_json,
        package_root=args.package_root,
        grounding_config=args.grounding_config,
        grounding_weights=args.grounding_weights,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
        use_dino=args.reward_mode != "outcome_only",
    )
    model, processor = load_model_processor(args)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rollout_log_path = out_dir / "rollouts.jsonl"
    recent_malformed: deque[bool] = deque(maxlen=max(10, args.rollout_n * 5))

    step = 0
    data_idx = 0
    try:
        with rollout_log_path.open("w", encoding="utf-8") as log_f:
            while step < args.max_steps:
                example = rows[data_idx % len(rows)]
                data_idx += 1
                completions = generate_rollouts(model, processor, example, args)
                scored = [scorer.score(example, completion, args.reward_mode) for completion in completions]
                rewards = torch.tensor([score["reward_total"] for score in scored], dtype=torch.float32, device=model_device(model))
                if rewards.numel() > 1 and float(rewards.std(unbiased=False)) > 1e-6:
                    advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)
                else:
                    advantages = rewards - rewards.mean()

                losses = []
                for completion, advantage in zip(completions, advantages):
                    logp = sequence_logprob(model, processor, example, completion, args.max_length)
                    losses.append(-advantage.detach() * logp)
                loss = torch.stack(losses).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()

                for info in scored:
                    recent_malformed.append(bool(info.get("malformed")))
                malformed_rate = sum(recent_malformed) / max(1, len(recent_malformed))
                row = {
                    "step": step,
                    "sample_id": example.get("sample_id"),
                    "reward_mode": args.reward_mode,
                    "loss": float(loss.detach().cpu()),
                    "reward_mean": float(rewards.mean().detach().cpu()),
                    "reward_std": float(rewards.std(unbiased=False).detach().cpu()),
                    "recent_malformed_rate": malformed_rate,
                    "rollouts": [
                        {"completion": completion, **info}
                        for completion, info in zip(completions, scored)
                    ],
                }
                log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                log_f.flush()
                if step % args.log_every == 0:
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "sample_id": row["sample_id"],
                                "loss": row["loss"],
                                "reward_mean": row["reward_mean"],
                                "reward_std": row["reward_std"],
                                "recent_malformed_rate": malformed_rate,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if len(recent_malformed) == recent_malformed.maxlen and malformed_rate > args.max_malformed_rate:
                    model.save_pretrained(out_dir)
                    processor.save_pretrained(out_dir)
                    raise RuntimeError(
                        f"Rollout validity collapsed: recent malformed rate {malformed_rate:.3f} "
                        f"> {args.max_malformed_rate:.3f}"
                    )
                step += 1
    finally:
        model.save_pretrained(out_dir)
        processor.save_pretrained(out_dir)

    summary = {
        "output_dir": str(out_dir),
        "reward_mode": args.reward_mode,
        "steps": step,
        "n_rows": len(rows),
        "rollout_log_path": str(rollout_log_path),
    }
    write_json(out_dir / "rl_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
