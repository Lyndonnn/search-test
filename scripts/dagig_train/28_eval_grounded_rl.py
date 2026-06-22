#!/usr/bin/env python3
"""Evaluate grounded SFT initializer and grounded RL variants on dev/test."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from grounded_pipeline_utils import load_jsonl, md_table, write_csv, write_json, write_jsonl


SCORER_PATH = Path(__file__).with_name("26_score_grounded_rollouts.py")
SPEC = importlib.util.spec_from_file_location("grounded_rollout_scorer", SCORER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not import scorer from {SCORER_PATH}")
SCORER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER_MODULE)
GroundedRewardScorer = SCORER_MODULE.GroundedRewardScorer


METHODS = [
    {
        "method": "ground_action_sft_initializer",
        "adapter_dir": "checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora",
        "train_reward_mode": "sft",
    },
    {
        "method": "rl_grounded_outcome_only_lowtemp_7b_lora",
        "adapter_dir": "checkpoints/dagig_rn03_10_grounded_rl/rl_grounded_outcome_only_lowtemp_7b_lora",
        "train_reward_mode": "outcome_only",
    },
    {
        "method": "rl_grounded_outcome_plus_ground_penalty_lowtemp_7b_lora",
        "adapter_dir": "checkpoints/dagig_rn03_10_grounded_rl/rl_grounded_outcome_plus_ground_penalty_lowtemp_7b_lora",
        "train_reward_mode": "outcome_plus_ground_penalty",
    },
    {
        "method": "rl_grounded_generic_process_lowtemp_7b_lora",
        "adapter_dir": "checkpoints/dagig_rn03_10_grounded_rl/rl_grounded_generic_process_lowtemp_7b_lora",
        "train_reward_mode": "generic_process",
    },
    {
        "method": "rl_grounded_dagig_lowtemp_7b_lora",
        "adapter_dir": "checkpoints/dagig_rn03_10_grounded_rl/rl_grounded_dagig_lowtemp_7b_lora",
        "train_reward_mode": "dagig_grounded",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/dagig_rn03_10_grounded_rl")
    parser.add_argument("--result_root", default="results/dagig_rn03_10_grounded_rl")
    parser.add_argument("--model_name_or_path", default="/data/zhengxiang/code/dagig/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--sft_adapter", default="checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora")
    parser.add_argument("--rl_ckpt_root", default="checkpoints/dagig_rn03_10_grounded_rl")
    parser.add_argument("--corpus_jsonl", default="data/dagig_rn03_10_retrieval/corpus.jsonl")
    parser.add_argument("--targets_json", default="data/dagig_rn03_10_retrieval/targets.json")
    parser.add_argument("--package_root", default="data/dagig_rn03_10_ground_expr_v3_full")
    parser.add_argument("--grounding_config", default=SCORER_MODULE.DINO_CONFIG)
    parser.add_argument("--grounding_weights", default=SCORER_MODULE.DINO_WEIGHTS)
    parser.add_argument("--box_threshold", type=float, default=0.10)
    parser.add_argument("--text_threshold", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--reuse_predictions", action="store_true")
    return parser.parse_args()


def method_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    specs = []
    for spec in METHODS:
        item = dict(spec)
        if item["method"] == "ground_action_sft_initializer":
            item["adapter_dir"] = args.sft_adapter
        else:
            item["adapter_dir"] = str(Path(args.rl_ckpt_root) / item["method"])
        specs.append(item)
    return specs


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


def generate_predictions(
    args: argparse.Namespace,
    method: str,
    adapter_dir: str,
    split_examples: dict[str, list[dict[str, Any]]],
    pred_dir: Path,
) -> list[Path]:
    output_paths = [pred_dir / f"{method}_{split}.jsonl" for split in args.splits]
    if args.reuse_predictions and all(path.is_file() for path in output_paths):
        return output_paths
    model, processor = load_model(args, adapter_dir)
    try:
        for split in args.splits:
            out_path = pred_dir / f"{method}_{split}.jsonl"
            total = len(split_examples[split])
            with out_path.open("w", encoding="utf-8") as f:
                for idx, example in enumerate(split_examples[split], start=1):
                    prediction = generate_one(model, processor, example, args.max_new_tokens)
                    row = {
                        "method": method,
                        "split": split,
                        "sample_id": example.get("sample_id"),
                        "prediction": prediction,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    if idx == 1 or idx % 10 == 0 or idx == total:
                        print(f"[generate] {method} {split} {idx}/{total}", flush=True)
    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output_paths


def rate(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [row.get(key) for row in rows if row.get(key) is not None]
    if not vals:
        return None
    return float(mean(float(v) for v in vals))


def aggregate(scored_rows: list[dict[str, Any]], method: str, split: str) -> dict[str, Any]:
    n = len(scored_rows)
    if n == 0:
        return {"method": method, "split": split, "n": 0}
    ious = [float(row.get("ground_iou") or 0.0) for row in scored_rows if row.get("ground_iou") is not None]
    return {
        "method": method,
        "split": split,
        "n": n,
        "format_valid": rate(scored_rows, "format_valid"),
        "malformed_rate": rate(scored_rows, "malformed"),
        "ground_non_empty": rate(scored_rows, "ground_non_empty"),
        "detection_rate": rate(scored_rows, "ground_detected"),
        "no_detection_rate": None if rate(scored_rows, "ground_detected") is None else 1.0 - float(rate(scored_rows, "ground_detected")),
        "mean_iou": float(mean(ious)) if ious else None,
        "iou_ge_0_1": sum(1 for row in scored_rows if (row.get("ground_iou") or 0.0) >= 0.1) / n,
        "iou_ge_0_3": sum(1 for row in scored_rows if (row.get("ground_iou") or 0.0) >= 0.3) / n,
        "iou_ge_0_5": sum(1 for row in scored_rows if (row.get("ground_iou") or 0.0) >= 0.5) / n,
        "center_hit": rate(scored_rows, "ground_center_hit"),
        "R@1": rate(scored_rows, "retrieval_r1"),
        "R@5": rate(scored_rows, "retrieval_r5"),
        "MRR": rate(scored_rows, "retrieval_mrr"),
        "query_anchor_hit": rate(scored_rows, "query_anchor_hit"),
        "query_specificity": rate(scored_rows, "query_specificity"),
        "evidence_support": rate(scored_rows, "evidence_support"),
        "unsupported_rate": rate(scored_rows, "unsupported_answer"),
        "spurious_success": rate(scored_rows, "spurious_success"),
        "EM": rate(scored_rows, "answer_em"),
        "F1": rate(scored_rows, "answer_f1"),
        "reward_total": rate(scored_rows, "reward_total"),
        "R_ground": rate(scored_rows, "R_ground"),
        "R_observe": rate(scored_rows, "R_observe"),
        "R_search": rate(scored_rows, "R_search"),
        "R_evidence": rate(scored_rows, "R_evidence"),
        "R_answer": rate(scored_rows, "R_answer"),
        "R_cost": rate(scored_rows, "R_cost"),
    }


def fmt_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def write_comparison_table(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "method",
        "split",
        "format_valid",
        "mean_iou",
        "iou_ge_0_3",
        "center_hit",
        "R@1",
        "R@5",
        "MRR",
        "query_anchor_hit",
        "evidence_support",
        "unsupported_rate",
        "spurious_success",
        "EM",
        "F1",
        "reward_total",
        "R_ground",
        "R_observe",
        "R_search",
        "R_evidence",
        "R_answer",
        "R_cost",
    ]
    display = [{key: fmt_value(row.get(key)) for key in columns} for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Grounded RL Comparison\n\n" + md_table(display, columns) + "\n", encoding="utf-8")


def find_row(rows: list[dict[str, Any]], method: str, split: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("method") == method and row.get("split") == split:
            return row
    return None


def improvement_notes(rows: list[dict[str, Any]], split: str) -> list[str]:
    dagig = find_row(rows, "rl_grounded_dagig_lowtemp_7b_lora", split)
    outcome = find_row(rows, "rl_grounded_outcome_only_lowtemp_7b_lora", split)
    generic = find_row(rows, "rl_grounded_generic_process_lowtemp_7b_lora", split)
    if not dagig or not outcome or not generic:
        return [f"{split}: missing one or more RL variants, cannot conclude DAG-IG comparison."]
    positive_keys = ["mean_iou", "iou_ge_0_3", "center_hit", "query_anchor_hit", "R@5", "MRR", "evidence_support", "F1", "EM"]
    lower_keys = ["malformed_rate", "unsupported_rate", "spurious_success"]
    better = []
    worse = []
    for key in positive_keys:
        d = dagig.get(key)
        base = max(float(outcome.get(key) or 0.0), float(generic.get(key) or 0.0))
        if d is not None and float(d) > base + 1e-6:
            better.append(key)
        elif d is not None and float(d) + 1e-6 < base:
            worse.append(key)
    for key in lower_keys:
        d = dagig.get(key)
        base = min(float(outcome.get(key) or 0.0), float(generic.get(key) or 0.0))
        if d is not None and float(d) < base - 1e-6:
            better.append(key)
        elif d is not None and float(d) > base + 1e-6:
            worse.append(key)
    if better:
        return [f"{split}: DAG-IG improves over both outcome-only and generic-process on {', '.join(better)}."]
    return [f"{split}: DAG-IG does not clearly beat both baselines on key process metrics; weaker keys: {', '.join(worse) or 'none'}."]


def write_report(args: argparse.Namespace, metric_rows: list[dict[str, Any]], missing_methods: list[dict[str, str]]) -> None:
    result_root = Path(args.result_root)
    table_path = result_root / "tables" / "grounded_rl_comparison.md"
    sft_dev = find_row(metric_rows, "ground_action_sft_initializer", "dev") or {}
    sft_test = find_row(metric_rows, "ground_action_sft_initializer", "test") or {}
    notes = []
    for split in args.splits:
        notes.extend(improvement_notes(metric_rows, split))
    collapsed = []
    for spec in method_specs(args):
        failed_marker = Path(args.rl_ckpt_root) / f"{spec['method']}.FAILED"
        if failed_marker.is_file():
            collapsed.append({"method": spec["method"], "marker": str(failed_marker)})
    reward_rows = [
        {
            "method": row.get("method"),
            "split": row.get("split"),
            "reward_total": fmt_value(row.get("reward_total")),
            "R_ground": fmt_value(row.get("R_ground")),
            "R_observe": fmt_value(row.get("R_observe")),
            "R_search": fmt_value(row.get("R_search")),
            "R_evidence": fmt_value(row.get("R_evidence")),
            "R_answer": fmt_value(row.get("R_answer")),
            "R_cost": fmt_value(row.get("R_cost")),
        }
        for row in metric_rows
    ]
    missing_text = "\n".join(f"- {m['method']}: missing `{m['adapter_dir']}`" for m in missing_methods) or "- none"
    collapsed_text = "\n".join(f"- {m['method']}: early-stopped/collapsed marker `{m['marker']}`" for m in collapsed) or "- none"
    report = [
        "# Grounded DAG-IG RL Report",
        "",
        "## 1. Starting Point",
        "",
        f"- SFT initializer: `{args.sft_adapter}`",
        "- Previous direct-bbox autonomous IoU>=0.3: about 2.65%.",
        "- Ground-action SFT + DINO had already established the grounded action interface: dev IoU>=0.3 about 42.86%, test about 45.31%.",
        f"- Re-evaluated SFT dev/test in this run: dev IoU>=0.3={fmt_value(sft_dev.get('iou_ge_0_3'))}, test IoU>=0.3={fmt_value(sft_test.get('iou_ge_0_3'))}.",
        "",
        "## 2. RL Data",
        "",
        "- Source: `data/dagig_rn03_10_grounded_rl/grounded_rl_train/dev/test.jsonl`.",
        "- Review-needed rows remain excluded because these files are built from the 620 hard-pass ground-action rows only.",
        "- Leakage-sensitive rows are retained for diagnostics and excluded from RL training reward computation by the trainer.",
        "",
        "## 3. Reward Definitions",
        "",
        "- `outcome_only`: answer EM/F1 with malformed penalty.",
        "- `outcome_plus_ground_penalty`: answer reward plus penalties for malformed/missing ground, DINO miss, extreme box, or missing search.",
        "- `generic_process`: format, non-empty ground, DINO detection/center proxy, query specificity, retrieval, evidence, and answer rewards without dependency gating.",
        "- `dagig_grounded`: dependency-aware components R_ground/R_observe/R_search/R_evidence/R_answer/R_cost; answer reward is gated by evidence support and search/observe rewards are gated by upstream grounded process quality.",
        "",
        "## 4. Low-Budget Setup",
        "",
        "- Initial adapter: grounded SFT LoRA.",
        "- Diagnostic budget: default 64 train samples, 20 steps, 2 rollouts/prompt, temperature 0.2.",
        "- This run is intended to test credit-assignment signal direction, not produce final paper-scale RL numbers.",
        "",
        "## 5. Missing Variants",
        "",
        missing_text,
        "",
        "## 5b. Collapsed / Partial Variants",
        "",
        collapsed_text,
        "",
        "Collapsed variants are shown only as diagnostics; they should not be treated as successful RL runs.",
        "",
        "## 6. Dev/Test Comparison",
        "",
        table_path.read_text(encoding="utf-8") if table_path.is_file() else "",
        "",
        "## 7. Reward Components",
        "",
        md_table(reward_rows),
        "",
        "## 8. Interpretation",
        "",
        *[f"- {note}" for note in notes],
        "- Overall: this low-budget run does not establish a clean DAG-IG win. Generic-process has stronger reward_total/evidence_support on dev and stronger reward_total/center-hit on test, while DAG-IG shows some search-side gains.",
        "- Answer EM remains very low across all methods; answer-level credit assignment is not solved by this diagnostic run.",
        "",
        "The grounded-interface evidence is already complete: replacing direct bbox generation with a ground-expression plus GroundingDINO interface substantially improves localization. The new RL evidence should be interpreted only as a low-budget diagnostic for long-horizon credit assignment.",
        "",
        "## 9. Failure Analysis",
        "",
        "- If DAG-IG improves process metrics but not EM/F1, the likely issue is downstream retrieval/answering rather than the grounding interface.",
        "- If malformed rate rises above baselines, the reward or rollout budget is not stabilizing the action grammar.",
        "- If grounding IoU drops while reward increases, the current proxies are too weak and need a stronger verifier or more explicit anti-spurious penalties.",
        "",
        "## 10. Next Steps",
        "",
        "- Run more steps and at least 3 seeds for any variant that improves process metrics here.",
        "- Add a stronger evidence verifier before claiming answer-level credit assignment.",
        "- Fix or review leakage-sensitive rows before scaling RL.",
        "- Keep direct-bbox as a negative localization baseline; it is not competitive with the grounded action interface.",
    ]
    (result_root / "GROUNDED_DAGIG_RL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    result_root = Path(args.result_root)
    eval_dir = result_root / "eval"
    pred_dir = eval_dir / "grounded_rl_predictions"
    table_dir = result_root / "tables"
    pred_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    split_examples: dict[str, list[dict[str, Any]]] = {}
    for split in args.splits:
        split_examples[split] = load_jsonl(Path(args.data_dir) / f"grounded_rl_{split}.jsonl", limit=args.limit)

    missing_methods = []
    jobs: list[dict[str, Any]] = []
    for spec in method_specs(args):
        adapter_dir = spec["adapter_dir"]
        if not Path(adapter_dir).is_dir():
            missing_methods.append({"method": spec["method"], "adapter_dir": adapter_dir})
            continue
        paths = generate_predictions(args, spec["method"], adapter_dir, split_examples, pred_dir)
        for split, path in zip(args.splits, paths):
            jobs.append({"method": spec["method"], "split": split, "path": path, "train_reward_mode": spec["train_reward_mode"]})

    scorer = GroundedRewardScorer(
        corpus_jsonl=args.corpus_jsonl,
        targets_json=args.targets_json,
        package_root=args.package_root,
        grounding_config=args.grounding_config,
        grounding_weights=args.grounding_weights,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
        use_dino=True,
    )
    metric_rows: list[dict[str, Any]] = []
    reward_component_rows: list[dict[str, Any]] = []
    for job in jobs:
        examples_by_id = {str(row.get("sample_id")): row for row in split_examples[job["split"]]}
        raw_rows = load_jsonl(job["path"])
        scored_rows = []
        total = len(raw_rows)
        for idx, raw in enumerate(raw_rows, start=1):
            example = examples_by_id[str(raw.get("sample_id"))]
            score = scorer.score(example, str(raw.get("prediction") or ""), "dagig_grounded")
            scored_rows.append({**raw, **score, "canonical_eval_reward_mode": "dagig_grounded", "train_reward_mode": job["train_reward_mode"]})
            if idx == 1 or idx % 20 == 0 or idx == total:
                print(f"[score] {job['method']} {job['split']} {idx}/{total}", flush=True)
        write_jsonl(job["path"], scored_rows)
        metrics = aggregate(scored_rows, job["method"], job["split"])
        metrics["train_reward_mode"] = job["train_reward_mode"]
        metric_rows.append(metrics)
        reward_component_rows.append(
            {
                "method": job["method"],
                "split": job["split"],
                "reward_total": metrics.get("reward_total"),
                "R_ground": metrics.get("R_ground"),
                "R_observe": metrics.get("R_observe"),
                "R_search": metrics.get("R_search"),
                "R_evidence": metrics.get("R_evidence"),
                "R_answer": metrics.get("R_answer"),
                "R_cost": metrics.get("R_cost"),
            }
        )

    dev_rows = [row for row in metric_rows if row.get("split") == "dev"]
    test_rows = [row for row in metric_rows if row.get("split") == "test"]
    write_csv(eval_dir / "grounded_rl_dev_metrics.csv", dev_rows)
    write_csv(eval_dir / "grounded_rl_test_metrics.csv", test_rows)
    write_csv(eval_dir / "grounded_rl_reward_components.csv", reward_component_rows)
    write_json(eval_dir / "grounded_rl_missing_methods.json", missing_methods)
    write_comparison_table(table_dir / "grounded_rl_comparison.md", metric_rows)
    write_report(args, metric_rows, missing_methods)
    print(json.dumps({"metrics": metric_rows, "missing_methods": missing_methods}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
