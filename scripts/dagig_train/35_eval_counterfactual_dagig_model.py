#!/usr/bin/env python3
"""Evaluate old baselines and the counterfactual DAG-IG preference model."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from grounded_pipeline_utils import load_jsonl, md_table, write_csv, write_jsonl


CF_SCORER_PATH = Path(__file__).with_name("30_score_counterfactual_dagig.py")
SPEC = importlib.util.spec_from_file_location("counterfactual_dagig_scorer", CF_SCORER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import scorer from {CF_SCORER_PATH}")
CF_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CF_MODULE)
CounterfactualDAGIGScorer = CF_MODULE.CounterfactualDAGIGScorer


METHODS = [
    {
        "method": "sft_initializer",
        "adapter": "checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora",
        "old_prediction_name": "ground_action_sft_initializer",
    },
    {
        "method": "outcome_only_lowbudget_rl",
        "adapter": "checkpoints/dagig_rn03_10_grounded_rl/rl_grounded_outcome_only_lowtemp_7b_lora",
        "old_prediction_name": "rl_grounded_outcome_only_lowtemp_7b_lora",
    },
    {
        "method": "outcome_plus_ground_penalty_lowbudget_rl",
        "adapter": "checkpoints/dagig_rn03_10_grounded_rl/rl_grounded_outcome_plus_ground_penalty_lowtemp_7b_lora",
        "old_prediction_name": "rl_grounded_outcome_plus_ground_penalty_lowtemp_7b_lora",
    },
    {
        "method": "generic_process_lowbudget_rl",
        "adapter": "checkpoints/dagig_rn03_10_grounded_rl/rl_grounded_generic_process_lowtemp_7b_lora",
        "old_prediction_name": "rl_grounded_generic_process_lowtemp_7b_lora",
    },
    {
        "method": "old_heuristic_dagig_lowbudget_rl",
        "adapter": "checkpoints/dagig_rn03_10_grounded_rl/rl_grounded_dagig_lowtemp_7b_lora",
        "old_prediction_name": "rl_grounded_dagig_lowtemp_7b_lora",
    },
    {
        "method": "counterfactual_dagig_rejection_sft",
        "adapter": "checkpoints/dagig_rn03_10_counterfactual_dagig/dagig_dpo_7b_lora",
        "old_prediction_name": "",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/dagig_rn03_10_grounded_rl")
    parser.add_argument("--counterfactual_dir", default="data/dagig_rn03_10_counterfactuals")
    parser.add_argument("--result_root", default="results/dagig_rn03_10_counterfactual_dagig")
    parser.add_argument("--old_prediction_root", default="results/dagig_rn03_10_grounded_rl/eval/grounded_rl_predictions")
    parser.add_argument("--model_name_or_path", default="/data/zhengxiang/code/dagig/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reuse_predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    parser.add_argument("--corpus_jsonl", default="data/dagig_rn03_10_retrieval/corpus.jsonl")
    parser.add_argument("--targets_json", default="data/dagig_rn03_10_retrieval/targets.json")
    parser.add_argument("--package_root", default="data/dagig_rn03_10_ground_expr_v3_full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_dino", action="store_true")
    return parser.parse_args()


def model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def build_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = str(example["prompt"])
    images = [p for p in example.get("images", []) if p]
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def load_model(args: argparse.Namespace, adapter_dir: str) -> tuple[Any, Any]:
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, processor


@torch.inference_mode()
def generate_one(model: Any, processor: Any, example: dict[str, Any], max_new_tokens: int) -> str:
    from qwen_vl_utils import process_vision_info

    messages = build_messages(example)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    device = model_device(model)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=getattr(processor.tokenizer, "eos_token_id", None),
    )
    new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def prediction_path(args: argparse.Namespace, method: dict[str, str], split: str) -> Path:
    if method.get("old_prediction_name"):
        old_path = Path(args.old_prediction_root) / f"{method['old_prediction_name']}_{split}.jsonl"
        if old_path.is_file():
            return old_path
    return Path(args.result_root) / "predictions" / f"{method['method']}_{split}.jsonl"


def jsonl_line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def complete_jsonl(path: Path, expected_rows: int) -> bool:
    return path.is_file() and jsonl_line_count(path) >= expected_rows


def ensure_predictions(args: argparse.Namespace, method: dict[str, str], examples_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    missing = []
    for split in args.splits:
        path = prediction_path(args, method, split)
        expected = len(examples_by_split[split])
        if args.reuse_predictions and complete_jsonl(path, expected):
            out[split] = path
        else:
            missing.append(split)
            out[split] = Path(args.result_root) / "predictions" / f"{method['method']}_{split}.jsonl"
    if not missing:
        return out
    adapter = method["adapter"]
    if not Path(adapter).is_dir():
        raise FileNotFoundError(f"Missing adapter for {method['method']}: {adapter}")
    model, processor = load_model(args, adapter)
    try:
        for split in missing:
            path = out[split]
            path.parent.mkdir(parents=True, exist_ok=True)
            total = len(examples_by_split[split])
            with path.open("w", encoding="utf-8") as f:
                for idx, example in enumerate(examples_by_split[split], start=1):
                    pred = generate_one(model, processor, example, args.max_new_tokens)
                    f.write(json.dumps({"method": method["method"], "split": split, "sample_id": example.get("sample_id"), "prediction": pred}, ensure_ascii=False) + "\n")
                    f.flush()
                    if idx == 1 or idx % 10 == 0 or idx == total:
                        print(f"[generate] {method['method']} {split} {idx}/{total}", flush=True)
    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def avg(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [row.get(key) for row in rows if row.get(key) is not None]
    if not vals:
        return None
    return float(mean(float(v) for v in vals))


def aggregate(rows: list[dict[str, Any]], method: str, split: str) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"method": method, "split": split, "n": 0}
    return {
        "method": method,
        "split": split,
        "n": n,
        "format_valid": avg(rows, "format_valid"),
        "iou_ge_0_3": sum(1 for row in rows if float(row.get("ground_iou") or 0.0) >= 0.3) / n,
        "center_hit": avg(rows, "ground_center_hit"),
        "query_anchor_hit": avg(rows, "query_anchor_hit"),
        "R@1": avg(rows, "retrieval_r1"),
        "R@5": avg(rows, "retrieval_r5"),
        "MRR": avg(rows, "retrieval_mrr"),
        "evidence_support": avg(rows, "evidence_support"),
        "unsupported_rate": avg(rows, "unsupported_answer"),
        "spurious_success": avg(rows, "spurious_success"),
        "EM": avg(rows, "answer_em"),
        "F1": avg(rows, "answer_f1"),
        "DAGIG_total": avg(rows, "DAGIG_total"),
        "R_ground": avg(rows, "R_ground"),
        "R_observe": avg(rows, "R_observe"),
        "R_search": avg(rows, "R_search"),
        "R_evidence": avg(rows, "R_evidence"),
        "R_answer": avg(rows, "R_answer"),
        "R_cost": avg(rows, "R_cost"),
    }


def write_report(args: argparse.Namespace, metrics: list[dict[str, Any]], missing: list[str]) -> None:
    result_root = Path(args.result_root)
    table_path = result_root / "tables" / "final_counterfactual_dagig_comparison.md"
    columns = [
        "method",
        "split",
        "format_valid",
        "iou_ge_0_3",
        "center_hit",
        "query_anchor_hit",
        "R@1",
        "R@5",
        "MRR",
        "evidence_support",
        "unsupported_rate",
        "spurious_success",
        "EM",
        "F1",
        "DAGIG_total",
        "R_ground",
        "R_observe",
        "R_search",
        "R_evidence",
        "R_answer",
        "R_cost",
    ]
    display = [
        {k: (f"{row.get(k):.4f}" if isinstance(row.get(k), float) else row.get(k)) for k in columns}
        for row in metrics
    ]
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("# Final Counterfactual DAG-IG Comparison\n\n" + md_table(display, columns) + "\n", encoding="utf-8")
    cf_rows = [row for row in metrics if row["method"] == "counterfactual_dagig_rejection_sft"]
    gen_rows = [row for row in metrics if row["method"] == "generic_process_lowbudget_rl"]
    notes = []
    for split in args.splits:
        cf = next((row for row in cf_rows if row["split"] == split), None)
        gen = next((row for row in gen_rows if row["split"] == split), None)
        if not cf:
            notes.append(f"- {split}: counterfactual DAG-IG preference model was not evaluated.")
        elif not gen:
            notes.append(f"- {split}: generic-process baseline missing, cannot claim improvement.")
        else:
            wins = []
            for key in ["query_anchor_hit", "R@5", "MRR", "evidence_support", "F1"]:
                if float(cf.get(key) or 0.0) > float(gen.get(key) or 0.0):
                    wins.append(key)
            lower_wins = []
            for key in ["unsupported_rate", "spurious_success"]:
                if float(cf.get(key) or 0.0) < float(gen.get(key) or 0.0):
                    lower_wins.append(key)
            if wins or lower_wins:
                notes.append(f"- {split}: counterfactual model improves over generic-process on {', '.join(wins + lower_wins)}.")
            else:
                notes.append(f"- {split}: counterfactual model does not beat generic-process on the requested process metrics.")
    missing_text = "\n".join(f"- {item}" for item in missing) or "- none"
    report = [
        "# Counterfactual DAG-IG Report",
        "",
        "## Scope",
        "",
        "This evaluates edge-level counterfactual DAG-IG scoring and the resulting rejection-SFT preference model. It does not treat reward_total as the only conclusion.",
        "",
        "## Missing Methods",
        "",
        missing_text,
        "",
        "## Final Comparison",
        "",
        table_path.read_text(encoding="utf-8"),
        "",
        "## Interpretation",
        "",
        *notes,
        "",
        "Do not claim success unless the counterfactual model is more stable than generic-process on query_anchor_hit, R@5/MRR, evidence_support, and unsupported/spurious metrics.",
    ]
    (result_root / "COUNTERFACTUAL_DAGIG_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    result_root = Path(args.result_root)
    scored_dir = result_root / "eval_scored"
    pred_dir = result_root / "predictions"
    scored_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    examples_by_split = {
        split: load_jsonl(Path(args.data_dir) / f"grounded_rl_{split}.jsonl", limit=args.limit)
        for split in args.splits
    }
    cfs_by_split = {
        split: {str(row.get("sample_id")): row for row in load_jsonl(Path(args.counterfactual_dir) / f"counterfactual_{split}.jsonl", limit=args.limit)}
        for split in args.splits
    }
    scorer = CounterfactualDAGIGScorer(
        corpus_jsonl=args.corpus_jsonl,
        targets_json=args.targets_json,
        package_root=args.package_root,
        device=args.device,
        use_dino=not args.no_dino,
    )
    metric_rows = []
    missing = []
    for method in METHODS:
        if not Path(method["adapter"]).is_dir():
            missing.append(f"{method['method']}: missing adapter `{method['adapter']}`")
            continue
        try:
            pred_paths = ensure_predictions(args, method, examples_by_split)
        except FileNotFoundError as exc:
            missing.append(str(exc))
            continue
        for split, path in pred_paths.items():
            examples = {str(row.get("sample_id")): row for row in examples_by_split[split]}
            raw_rows = load_jsonl(path)
            out_path = scored_dir / f"{method['method']}_{split}.jsonl"
            if args.reuse_predictions and complete_jsonl(out_path, len(raw_rows)):
                scored = load_jsonl(out_path)
                metric_rows.append(aggregate(scored, method["method"], split))
                print(f"[reuse scored] {method['method']} {split} {len(scored)} rows", flush=True)
                continue
            scored = []
            total = len(raw_rows)
            for idx, raw in enumerate(raw_rows, start=1):
                sample_id = str(raw.get("sample_id"))
                example = examples.get(sample_id)
                cf = cfs_by_split[split].get(sample_id)
                if example is None or cf is None:
                    continue
                score = scorer.score(example, cf, str(raw.get("prediction") or ""))
                scored.append({"method": method["method"], "split": split, "sample_id": sample_id, "prediction": raw.get("prediction"), **score})
                if idx == 1 or idx % 20 == 0 or idx == total:
                    print(f"[score] {method['method']} {split} {idx}/{total}", flush=True)
            write_jsonl(out_path, scored)
            metric_rows.append(aggregate(scored, method["method"], split))
    write_csv(result_root / "tables" / "final_counterfactual_dagig_comparison.csv", metric_rows)
    write_report(args, metric_rows, missing)
    print(json.dumps({"metrics": metric_rows, "missing": missing}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
