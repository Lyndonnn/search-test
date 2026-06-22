#!/usr/bin/env python3
"""Minimal offline GRPO-style LoRA training for Pix2Fact-DAGIG.

This is a compact research loop for the fixed offline retrieval setting. It is
deliberately simple and logs every rollout so reward variants can be audited.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


SEGMENTS = ["observe", "search_decision", "search", "evidence", "answer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--init_adapter_dir", default="")
    parser.add_argument("--corpus_jsonl", default="data/dagig_retrieval/corpus.jsonl")
    parser.add_argument("--targets_json", default="data/dagig_retrieval/targets.json")
    parser.add_argument("--reward_mode", choices=["outcome_only", "outcome_plus_search_penalty", "generic_process", "text_ig", "dagig"], default="dagig")
    parser.add_argument("--rollout_n", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args()


def load_jsonl(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
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
    pc: dict[str, int] = defaultdict(int)
    tc: dict[str, int] = defaultdict(int)
    for tok in pred_tokens:
        pc[tok] += 1
    for tok in target_tokens:
        tc[tok] += 1
    common = sum(min(pc[tok], tc[tok]) for tok in pc)
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
    return match.group(1).strip() if match else ""


def extract_segments(text: str) -> dict[str, str]:
    return {segment: extract_tag(text, segment) for segment in SEGMENTS}


def query_anchor_hit(query: str, anchor: str) -> bool:
    q_tokens = set(tokens(query))
    anchor_tokens = [tok for tok in tokens(anchor) if len(tok) > 1]
    return bool(anchor_tokens and any(tok in q_tokens for tok in anchor_tokens))


class BM25:
    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = docs
        self.doc_tokens = [tokens(str(doc.get("text", ""))) for doc in docs]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        n_docs = len(docs)
        self.idf = {tok: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5)) for tok, freq in df.items()}

    def rank(self, query: str) -> list[tuple[str, float]]:
        q_counts = Counter(tokens(query))
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
        return ranked


def reciprocal_rank(ranked: list[tuple[str, float]], targets: set[str]) -> float:
    for idx, (doc_id, _score) in enumerate(ranked, start=1):
        if doc_id in targets:
            return 1.0 / idx
    return 0.0


def score_reward(example: dict[str, Any], prediction: str, bm25: BM25, targets: dict[str, list[str]], mode: str) -> tuple[float, dict[str, Any]]:
    pred = extract_segments(prediction)
    target = example.get("target_segments") if isinstance(example.get("target_segments"), dict) else {}
    target_ids = set(targets.get(str(example.get("sample_id", "")), []))
    ranked = bm25.rank(pred["search"])
    top_ids = [doc_id for doc_id, _score in ranked]
    r1 = 1.0 if target_ids and bool(set(top_ids[:1]) & target_ids) else 0.0
    r5 = 1.0 if target_ids and bool(set(top_ids[:5]) & target_ids) else 0.0
    mrr = reciprocal_rank(ranked, target_ids)
    answer_em = 1.0 if contains_match(pred["answer"], str(target.get("answer") or example.get("answer") or "")) else 0.0
    answer_f1 = token_f1(pred["answer"], str(target.get("answer") or example.get("answer") or ""))
    observe_f1 = token_f1(pred["observe"], str(target.get("observe", "")))
    query_f1 = token_f1(pred["search"], str(target.get("search", "")))
    evidence_f1 = token_f1(pred["evidence"], str(target.get("evidence", "")))
    anchor_hit = 1.0 if query_anchor_hit(pred["search"], str(example.get("visual_anchor", ""))) else 0.0
    valid_format = 1.0 if pred["observe"].strip() and pred["search"].strip() and pred["answer"].strip() else 0.0
    search_call = 1.0 if pred["search"].strip() else 0.0
    search_needed = 1.0 if bool(example.get("search_needed", True)) else 0.0
    missing_search = 1.0 if search_needed and not search_call else 0.0
    unnecessary_search = 1.0 if search_call and not search_needed else 0.0
    spurious = 1.0 if answer_em and (anchor_hit < 1.0 or r5 < 1.0 or evidence_f1 < 0.20) else 0.0

    outcome_only = answer_em
    search_penalty = max(0.0, answer_em - 0.05 * search_call - 0.25 * missing_search)
    generic = 0.20 * valid_format + 0.20 * observe_f1 + 0.20 * min(1.0, len(tokens(pred["search"])) / 8.0) + 0.20 * r5 + 0.20 * answer_f1
    text_ig = 0.35 * query_f1 + 0.35 * mrr + 0.30 * answer_f1
    observe_reward = observe_f1 if anchor_hit or observe_f1 >= 0.35 else 0.25 * observe_f1
    search_reward = 0.50 * anchor_hit + 0.30 * r5 + 0.20 * mrr
    evidence_reward = max(evidence_f1, r1)
    answer_reward = answer_f1 * (0.5 + 0.5 * max(evidence_f1, r5))
    cost = 0.05 * search_call + 0.20 * unnecessary_search + 0.30 * missing_search + 0.30 * spurious
    dagig = max(0.0, 0.10 + 0.20 * observe_reward + 0.25 * search_reward + 0.20 * evidence_reward + 0.25 * answer_reward - cost)
    rewards = {
        "outcome_only": outcome_only,
        "outcome_plus_search_penalty": search_penalty,
        "generic_process": generic,
        "text_ig": text_ig,
        "dagig": dagig,
    }
    info = {
        "prediction": pred,
        "answer_em": answer_em,
        "answer_f1": answer_f1,
        "observe_f1": observe_f1,
        "query_f1": query_f1,
        "evidence_f1": evidence_f1,
        "query_anchor_hit": anchor_hit,
        "retrieval_r1": r1,
        "retrieval_r5": r5,
        "retrieval_mrr": mrr,
        "valid_format": valid_format,
        "spurious_success": spurious,
        "reward": rewards[mode],
        "reward_mode": mode,
    }
    return float(rewards[mode]), info


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

    from peft import LoraConfig, PeftModel, get_peft_model

    if args.init_adapter_dir:
        model = PeftModel.from_pretrained(model, args.init_adapter_dir, is_trainable=True)
    else:
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, config)
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
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.rollout_n,
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
    model_inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in model_inputs.items()}
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rows = load_jsonl(args.train_file, args.limit)
    if not rows:
        raise ValueError(f"No training rows loaded from {args.train_file}")
    corpus = load_jsonl(args.corpus_jsonl)
    with Path(args.targets_json).open("r", encoding="utf-8") as f:
        targets = json.load(f)
    bm25 = BM25(corpus)
    os.makedirs(args.output_dir, exist_ok=True)
    rollout_log_path = Path(args.output_dir) / "rollouts.jsonl"
    with (Path(args.output_dir) / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args) | {"n_rows": len(rows)}, f, ensure_ascii=False, indent=2)

    model, processor = load_model_processor(args)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    step = 0
    data_idx = 0
    with rollout_log_path.open("w", encoding="utf-8") as log_f:
        while step < args.max_steps:
            example = rows[data_idx % len(rows)]
            data_idx += 1
            completions = generate_rollouts(model, processor, example, args)
            scored = [score_reward(example, completion, bm25, targets, args.reward_mode) for completion in completions]
            rewards = torch.tensor([score for score, _info in scored], dtype=torch.float32, device=model.device)
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

            row = {
                "step": step,
                "sample_id": example.get("sample_id"),
                "reward_mode": args.reward_mode,
                "loss": float(loss.detach().cpu()),
                "reward_mean": float(rewards.mean().detach().cpu()),
                "reward_std": float(rewards.std(unbiased=False).detach().cpu()),
                "rollouts": [
                    {"completion": completion, **info}
                    for completion, (_score, info) in zip(completions, scored)
                ],
            }
            log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_f.flush()
            if step % args.log_every == 0:
                print(json.dumps({k: row[k] for k in ["step", "sample_id", "loss", "reward_mean", "reward_std"]}, ensure_ascii=False))
            step += 1

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"saved RL LoRA adapter to {args.output_dir}")
    print(f"wrote {rollout_log_path}")


if __name__ == "__main__":
    main()
