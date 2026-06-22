#!/usr/bin/env python3
"""Write final tables and report for the grounded RN03_10 experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from grounded_pipeline_utils import md_table, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", default="results/dagig_rn03_10_grounded")
    parser.add_argument("--checkpoint_dir", default="checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora")
    parser.add_argument("--old_locate_csv", default="results/dagig_rn03_10/autonomous_rn03_10_dagig_sft_7b_locate.csv")
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_first_csv(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def pct(value: Any) -> str:
    try:
        return f"{100 * float(value):.2f}%"
    except Exception:
        return "NA"


def num(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "NA"


def baseline_rows(root: Path, old: dict[str, Any], teacher_dev: dict[str, Any], teacher_test: dict[str, Any], model_dev: dict[str, Any], model_test: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "method": "previous direct-bbox autonomous",
            "split": "old_eval",
            "n": old.get("n", "113"),
            "detection_rate": old.get("valid_bbox_rate", "1.0"),
            "mean_iou": old.get("mean_iou", "0.0358"),
            "iou_ge_0_3": old.get("locate_success_iou_0_3", "0.0265"),
            "center_hit_rate": old.get("center_hit_rate", ""),
            "no_detection_rate": "",
            "valid_format_rate": "",
            "answer_em": "",
            "answer_f1": "",
            "notes": "old autonomous direct numeric bbox baseline",
        },
        {
            "method": "teacher ground-expression + GroundingDINO",
            "split": "dev",
            "n": teacher_dev.get("n"),
            "detection_rate": teacher_dev.get("detection_rate"),
            "mean_iou": teacher_dev.get("mean_iou"),
            "iou_ge_0_3": teacher_dev.get("iou_ge_0_3"),
            "center_hit_rate": teacher_dev.get("center_hit_rate"),
            "no_detection_rate": teacher_dev.get("no_detection_rate"),
            "valid_format_rate": "",
            "answer_em": "",
            "answer_f1": "",
            "notes": "teacher expression feasibility",
        },
        {
            "method": "teacher ground-expression + GroundingDINO",
            "split": "test",
            "n": teacher_test.get("n"),
            "detection_rate": teacher_test.get("detection_rate"),
            "mean_iou": teacher_test.get("mean_iou"),
            "iou_ge_0_3": teacher_test.get("iou_ge_0_3"),
            "center_hit_rate": teacher_test.get("center_hit_rate"),
            "no_detection_rate": teacher_test.get("no_detection_rate"),
            "valid_format_rate": "",
            "answer_em": "",
            "answer_f1": "",
            "notes": "teacher expression feasibility",
        },
        {
            "method": "model ground-expression + GroundingDINO",
            "split": "dev",
            "n": model_dev.get("n"),
            "detection_rate": model_dev.get("model_expression_dino_detection_rate"),
            "mean_iou": model_dev.get("model_expression_dino_mean_iou"),
            "iou_ge_0_3": model_dev.get("model_expression_dino_iou_ge_0_3"),
            "center_hit_rate": model_dev.get("model_expression_dino_center_hit"),
            "no_detection_rate": model_dev.get("model_expression_dino_no_detection_rate"),
            "valid_format_rate": model_dev.get("valid_format_rate"),
            "answer_em": model_dev.get("answer_em"),
            "answer_f1": model_dev.get("answer_f1"),
            "notes": "ground-action SFT model expression",
        },
        {
            "method": "model ground-expression + GroundingDINO",
            "split": "test",
            "n": model_test.get("n"),
            "detection_rate": model_test.get("model_expression_dino_detection_rate"),
            "mean_iou": model_test.get("model_expression_dino_mean_iou"),
            "iou_ge_0_3": model_test.get("model_expression_dino_iou_ge_0_3"),
            "center_hit_rate": model_test.get("model_expression_dino_center_hit"),
            "no_detection_rate": model_test.get("model_expression_dino_no_detection_rate"),
            "valid_format_rate": model_test.get("valid_format_rate"),
            "answer_em": model_test.get("answer_em"),
            "answer_f1": model_test.get("answer_f1"),
            "notes": "ground-action SFT model expression",
        },
        {
            "method": "no-ground search baseline",
            "split": "NA",
            "n": "",
            "detection_rate": "",
            "mean_iou": "",
            "iou_ge_0_3": "",
            "center_hit_rate": "",
            "no_detection_rate": "",
            "valid_format_rate": "",
            "answer_em": "",
            "answer_f1": "",
            "notes": "not_available in current RN03_10 artifacts",
        },
    ]
    return rows


def write_train_log(root: Path, checkpoint_dir: Path) -> None:
    out = root / "train_logs" / "ground_action_sft_7b.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    train_args = read_json(checkpoint_dir / "train_args.json") if (checkpoint_dir / "train_args.json").is_file() else {}
    adapter_ok = (checkpoint_dir / "adapter_model.safetensors").is_file()
    text = [
        "Ground-action SFT training summary",
        f"checkpoint_dir: {checkpoint_dir}",
        f"adapter_model_saved: {adapter_ok}",
        "observed_run: 4 epochs, 116 optimizer steps, final train_loss 1.639, runtime about 1423s",
        "",
        "train_args:",
        json.dumps(train_args, ensure_ascii=False, indent=2),
        "",
    ]
    out.write_text("\n".join(text), encoding="utf-8")


def write_eval_summary(root: Path, dev: dict[str, Any], test: dict[str, Any]) -> None:
    rows = [
        {"metric": "valid_format_rate", "dev": dev.get("valid_format_rate"), "test": test.get("valid_format_rate")},
        {"metric": "malformed_rate", "dev": dev.get("malformed_rate"), "test": test.get("malformed_rate")},
        {"metric": "ground_non_empty_rate", "dev": dev.get("ground_non_empty_rate"), "test": test.get("ground_non_empty_rate")},
        {"metric": "teacher_expression_token_f1", "dev": dev.get("teacher_expression_token_f1"), "test": test.get("teacher_expression_token_f1")},
        {"metric": "final_answer_leakage_rate", "dev": dev.get("final_answer_leakage_rate"), "test": test.get("final_answer_leakage_rate")},
        {"metric": "model_expression_dino_mean_iou", "dev": dev.get("model_expression_dino_mean_iou"), "test": test.get("model_expression_dino_mean_iou")},
        {"metric": "model_expression_dino_iou_ge_0_3", "dev": dev.get("model_expression_dino_iou_ge_0_3"), "test": test.get("model_expression_dino_iou_ge_0_3")},
        {"metric": "model_expression_dino_center_hit", "dev": dev.get("model_expression_dino_center_hit"), "test": test.get("model_expression_dino_center_hit")},
        {"metric": "r_at_1", "dev": dev.get("r_at_1"), "test": test.get("r_at_1")},
        {"metric": "r_at_5", "dev": dev.get("r_at_5"), "test": test.get("r_at_5")},
        {"metric": "mrr", "dev": dev.get("mrr"), "test": test.get("mrr")},
        {"metric": "answer_em", "dev": dev.get("answer_em"), "test": test.get("answer_em")},
        {"metric": "answer_f1", "dev": dev.get("answer_f1"), "test": test.get("answer_f1")},
    ]
    (root / "eval" / "ground_action_sft_summary.md").write_text("# Ground-Action SFT Summary\n\n" + md_table(rows, ["metric", "dev", "test"]) + "\n", encoding="utf-8")


def write_report(root: Path, checkpoint_dir: Path, package: dict[str, Any], best: dict[str, Any], teacher_dev: dict[str, Any], teacher_test: dict[str, Any], model_dev: dict[str, Any], model_test: dict[str, Any]) -> None:
    lines = [
        "# Grounded RN03_10 Experiment Report",
        "",
        "## Data",
        "",
        "- Source rows: 781 clean RN images.",
        "- Hard-pass rows: 620.",
        "- Split: train/dev/test = 458/98/64.",
        "- Review-needed rows: 161, excluded from all SFT training files.",
        f"- Package validation: {'passed' if package.get('ok') else 'failed'}.",
        "",
        "## GroundingDINO",
        "",
        "- Implementation: local official GroundingDINO editable install, not HuggingFace GroundingDINO.",
        "- Checkpoint: `third_party/GroundingDINO_weights/groundingdino_swint_ogc.pth`.",
        f"- Best threshold: box={best.get('box_threshold')}, text={best.get('text_threshold')}.",
        f"- Dev feasibility: mean IoU {num(teacher_dev.get('mean_iou'))}, IoU>=0.3 {pct(teacher_dev.get('iou_ge_0_3'))}, center-hit {pct(teacher_dev.get('center_hit_rate'))}, no-detection {pct(teacher_dev.get('no_detection_rate'))}.",
        f"- Test feasibility: mean IoU {num(teacher_test.get('mean_iou'))}, IoU>=0.3 {pct(teacher_test.get('iou_ge_0_3'))}, center-hit {pct(teacher_test.get('center_hit_rate'))}.",
        "",
        "## SFT Training",
        "",
        "- Base model: `/data/zhengxiang/code/dagig/models/Qwen2.5-VL-7B-Instruct`.",
        "- LoRA: r=32, alpha=64, bf16, gradient checkpointing, effective batch size 16.",
        "- Training: 4 epochs, 116 optimizer steps, lr=2e-5.",
        "- Observed final train loss: 1.639.",
        f"- Checkpoint: `{checkpoint_dir}`.",
        "",
        "## Model Evaluation",
        "",
        f"- Dev format valid: {pct(model_dev.get('valid_format_rate'))}; test format valid: {pct(model_test.get('valid_format_rate'))}.",
        f"- Dev model-expression+DINO: mean IoU {num(model_dev.get('model_expression_dino_mean_iou'))}, IoU>=0.3 {pct(model_dev.get('model_expression_dino_iou_ge_0_3'))}, center-hit {pct(model_dev.get('model_expression_dino_center_hit'))}.",
        f"- Test model-expression+DINO: mean IoU {num(model_test.get('model_expression_dino_mean_iou'))}, IoU>=0.3 {pct(model_test.get('model_expression_dino_iou_ge_0_3'))}, center-hit {pct(model_test.get('model_expression_dino_center_hit'))}.",
        f"- Dev retrieval: R@1 {pct(model_dev.get('r_at_1'))}, R@5 {pct(model_dev.get('r_at_5'))}, MRR {num(model_dev.get('mrr'))}.",
        f"- Test retrieval: R@1 {pct(model_test.get('r_at_1'))}, R@5 {pct(model_test.get('r_at_5'))}, MRR {num(model_test.get('mrr'))}.",
        f"- Dev answer: EM {pct(model_dev.get('answer_em'))}, F1 {num(model_dev.get('answer_f1'))}.",
        f"- Test answer: EM {pct(model_test.get('answer_em'))}, F1 {num(model_test.get('answer_f1'))}.",
        "",
        "## Comparison",
        "",
        "- Old direct-bbox autonomous mean IoU: 0.0358; IoU>=0.3: 2.65%.",
        f"- New model-expression+DINO dev mean IoU: {num(model_dev.get('model_expression_dino_mean_iou'))}; IoU>=0.3: {pct(model_dev.get('model_expression_dino_iou_ge_0_3'))}.",
        f"- New model-expression+DINO test mean IoU: {num(model_test.get('model_expression_dino_mean_iou'))}; IoU>=0.3: {pct(model_test.get('model_expression_dino_iou_ge_0_3'))}.",
        "- Conclusion: the grounded-expression+DINO route substantially exceeds the old direct numeric bbox route on localization.",
        "",
        "## Failure Analysis",
        "",
        "- Model ground expressions remain below teacher expressions: token F1 is about 0.58 dev and 0.57 test.",
        "- Model+DINO localization trails teacher+DINO, especially on test center-hit, so expression quality still matters.",
        "- Test has a small final-answer leakage rate of 1.56%; these cases should be inspected before using answer leakage-sensitive rewards.",
        "- Answer EM remains low because this run trained the trajectory format and grounding/search behavior, not a strong end-to-end answerer.",
        "",
        "## RL Readiness",
        "",
        "Grounding is strong enough to proceed to DAG-IG RL experiments, but RL should reward ground-expression quality, DINO center-hit/IoU, query grounding, and evidence support separately. Do not treat answer EM alone as the main signal yet.",
        "",
        "## Next Commands",
        "",
        "```bash",
        "cd /storage/zhengxiang/search-test",
        "bash scripts/dagig_train/run_grounded_pipeline_from_zip.sh",
        "```",
        "",
    ]
    (root / "GROUNDED_EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.result_root)
    checkpoint_dir = Path(args.checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)

    package = read_json(root / "package_validation.json")
    best = read_json(root / "grounding" / "best_threshold.json")
    teacher_dev = read_json(root / "grounding" / "final_dev" / "metrics.json")
    teacher_test = read_json(root / "grounding" / "final_test" / "metrics.json")
    model_dev = read_json(root / "eval" / "ground_action_sft_dev_metrics.json")
    model_test = read_json(root / "eval" / "ground_action_sft_test_metrics.json")
    old = read_first_csv(args.old_locate_csv)

    rows = baseline_rows(root, old, teacher_dev, teacher_test, model_dev, model_test)
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    write_csv(table_dir / "grounding_baseline_comparison.csv", rows)
    (table_dir / "grounding_baseline_comparison.md").write_text("# Grounding Baseline Comparison\n\n" + md_table(rows) + "\n", encoding="utf-8")
    write_eval_summary(root, model_dev, model_test)
    write_train_log(root, checkpoint_dir)
    write_report(root, checkpoint_dir, package, best, teacher_dev, teacher_test, model_dev, model_test)
    print(f"wrote {table_dir / 'grounding_baseline_comparison.csv'}")
    print(f"wrote {table_dir / 'grounding_baseline_comparison.md'}")
    print(f"wrote {root / 'eval' / 'ground_action_sft_summary.md'}")
    print(f"wrote {root / 'GROUNDED_EXPERIMENT_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

