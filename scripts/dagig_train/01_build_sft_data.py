#!/usr/bin/env python3
"""Build weighted SFT ablation datasets from DAG-IG Pix2Fact reward labels."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


VARIANTS = ["outcome_only", "local_ig", "dagig"]
TASKS = ["local_observation", "search_query", "answer_from_evidence"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package_dir", required=True)
    parser.add_argument("--out_dir", default="data")
    parser.add_argument(
        "--main_file",
        default="dagig_relabel/qwen_dagig_reward_labeled_30_with_image_paths.jsonl",
        help="Path relative to package_dir unless absolute.",
    )
    parser.add_argument("--repeat_scale", type=float, default=4.0)
    parser.add_argument("--min_answer_weight", type=float, default=0.2)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected object at {path}:{line_no}, got {type(row)}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path} n={len(rows)}")


def resolve_path(package_dir: Path, rel_or_abs: str) -> str:
    path = Path(str(rel_or_abs))
    if path.is_absolute():
        return str(path)
    return str((package_dir / path).resolve())


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def bounded_components(row: dict[str, Any]) -> dict[str, float]:
    comp = row.get("bounded_components") or {}
    if not isinstance(comp, dict):
        comp = {}
    return {
        "future_r": max(0.0, safe_float(comp.get("future_r"))),
        "direct_r": max(0.0, safe_float(comp.get("direct_r"))),
        "evidence_r": max(0.0, safe_float(comp.get("evidence_r"))),
    }


def variant_weights(row: dict[str, Any], min_answer_weight: float) -> dict[str, dict[str, float]]:
    comp = bounded_components(row)
    future_r = comp["future_r"]
    direct_r = comp["direct_r"]
    evidence_r = comp["evidence_r"]
    return {
        "outcome_only": {
            "local_observation": 0.0,
            "search_query": 0.0,
            "answer_from_evidence": 1.0,
        },
        "local_ig": {
            "local_observation": direct_r,
            "search_query": 0.0,
            "answer_from_evidence": max(min_answer_weight, evidence_r),
        },
        "dagig": {
            "local_observation": 0.30 * direct_r + 0.70 * future_r,
            "search_query": future_r,
            "answer_from_evidence": max(min_answer_weight, evidence_r),
        },
    }


def repeat_count(loss_weight: float, repeat_scale: float) -> int:
    if loss_weight <= 0:
        return 0
    return max(1, int(round(loss_weight * repeat_scale)))


def task_prompt(task_type: str, row: dict[str, Any]) -> str:
    question = str(row.get("question", "")).strip()
    local = str(row.get("qwen_local_observation", "")).strip()
    evidence = str(row.get("selected_evidence_text", "")).strip()
    if task_type == "local_observation":
        return (
            "Look at the image crop and identify the key local visual clue needed to answer the question. "
            "Return a short phrase only. Do not answer the final question.\n\n"
            f"Question: {question}"
        )
    if task_type == "search_query":
        return (
            "Given the question and local visual observation, write a concise web search query. "
            "Return only the query.\n\n"
            f"Question: {question}\nLocal visual observation: {local}"
        )
    if task_type == "answer_from_evidence":
        return (
            "Answer the question using the provided evidence. Return the shortest final answer only.\n\n"
            f"Question: {question}\nEvidence: {evidence}"
        )
    raise ValueError(f"Unsupported task_type={task_type}")


def task_target(task_type: str, row: dict[str, Any]) -> str:
    key = {
        "local_observation": "qwen_local_observation",
        "search_query": "qwen_search_query",
        "answer_from_evidence": "answer_target",
    }[task_type]
    return str(row.get(key, "")).strip()


def task_images(task_type: str, row: dict[str, Any], package_dir: Path) -> dict[str, Any]:
    image_paths = row.get("package_image_paths") or {}
    if not isinstance(image_paths, dict):
        image_paths = {}
    if task_type != "local_observation":
        return {"images": [], "image_path": "", "full_image_path": ""}
    crop = image_paths.get("qwen_crop_readable") or image_paths.get("crop_fixed")
    full = image_paths.get("qwen_full_resized") or image_paths.get("full_original")
    if not crop:
        raise FileNotFoundError(f"Missing crop path for sample_id={row.get('sample_id')}")
    crop_abs = resolve_path(package_dir, str(crop))
    full_abs = resolve_path(package_dir, str(full)) if full else ""
    if not Path(crop_abs).is_file():
        raise FileNotFoundError(f"Crop image missing for sample_id={row.get('sample_id')}: {crop_abs}")
    if full_abs and not Path(full_abs).is_file():
        raise FileNotFoundError(f"Full image missing for sample_id={row.get('sample_id')}: {full_abs}")
    images = [crop_abs]
    if full_abs:
        images.append(full_abs)
    return {"images": images, "image_path": crop_abs, "full_image_path": full_abs}


def paper_use(task_type: str, variant: str) -> str:
    if task_type == "local_observation":
        return f"{variant}: tests whether training credits local visual grounding."
    if task_type == "search_query":
        return f"{variant}: tests whether training credits future search action generation."
    return f"{variant}: tests final answer from supporting evidence."


def make_example(
    row: dict[str, Any],
    variant: str,
    task_type: str,
    loss_weight: float,
    package_dir: Path,
    repeat_scale: float,
) -> dict[str, Any]:
    image_info = task_images(task_type, row, package_dir)
    prompt = task_prompt(task_type, row)
    target = task_target(task_type, row)
    if not target:
        raise ValueError(f"Empty target for sample_id={row.get('sample_id')} task={task_type}")
    return {
        "sample_id": row.get("sample_id"),
        "variant": variant,
        "task_type": task_type,
        "prompt": prompt,
        "target": target,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        **image_info,
        "loss_weight": float(loss_weight),
        "repeat_count": repeat_count(float(loss_weight), repeat_scale),
        "reward_components": bounded_components(row),
        "raw_ig": row.get("raw_ig", {}),
        "reward_variants": row.get("reward_variants", {}),
        "final_seed_source": row.get("final_seed_source"),
        "audit_severity": row.get("audit_severity"),
        "paper_use": paper_use(task_type, variant),
    }


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [r for r in rows if r.get("final_seed_source") == "original_full_exec"]
    dev = [r for r in rows if r.get("final_seed_source") == "strict_local_retry_recovered"]
    if not train or not dev:
        raise ValueError(
            "Expected non-empty train/dev split by final_seed_source "
            "(original_full_exec / strict_local_retry_recovered)."
        )
    return train, dev


def build_examples(rows: list[dict[str, Any]], package_dir: Path, repeat_scale: float, min_answer_weight: float) -> dict[str, list[dict[str, Any]]]:
    out = {variant: [] for variant in VARIANTS}
    for row in rows:
        weights = variant_weights(row, min_answer_weight)
        for variant in VARIANTS:
            for task_type in TASKS:
                ex = make_example(
                    row=row,
                    variant=variant,
                    task_type=task_type,
                    loss_weight=weights[variant][task_type],
                    package_dir=package_dir,
                    repeat_scale=repeat_scale,
                )
                out[variant].append(ex)
    return out


def expanded(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for ex in examples:
        for i in range(int(ex.get("repeat_count", 0))):
            new_ex = dict(ex)
            new_ex["repeat_index"] = i
            rows.append(new_ex)
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]], examples_by_variant: dict[str, list[dict[str, Any]]]) -> None:
    summary: dict[str, Any] = {
        "input_n": len(rows),
        "source_counts": dict(Counter(str(r.get("final_seed_source")) for r in rows)),
        "audit_counts": dict(Counter(str(r.get("audit_severity")) for r in rows)),
        "variants": {},
    }
    for variant, examples in examples_by_variant.items():
        summary["variants"][variant] = {
            "n_examples": len(examples),
            "n_expanded": len(expanded(examples)),
            "task_counts": dict(Counter(ex["task_type"] for ex in examples)),
            "positive_weight_task_counts": dict(Counter(ex["task_type"] for ex in examples if ex["loss_weight"] > 0)),
            "loss_weight_mean": sum(float(ex["loss_weight"]) for ex in examples) / len(examples) if examples else 0.0,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    package_dir = Path(args.package_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    main_path = Path(args.main_file)
    if not main_path.is_absolute():
        main_path = package_dir / main_path
    if not main_path.is_file():
        raise FileNotFoundError(f"Training file missing: {main_path}")

    rows = load_jsonl(main_path)
    train_rows, dev_rows = split_rows(rows)
    examples_by_variant = build_examples(train_rows, package_dir, args.repeat_scale, args.min_answer_weight)

    out_dir.mkdir(parents=True, exist_ok=True)
    for variant, examples in examples_by_variant.items():
        write_jsonl(out_dir / f"sft_{variant}.jsonl", examples)
        write_jsonl(out_dir / f"sft_{variant}_expanded.jsonl", expanded(examples))

    eval_clean_dev = []
    eval_train_sanity = []
    for variant in VARIANTS:
        eval_clean_dev.extend(build_examples(dev_rows, package_dir, args.repeat_scale, args.min_answer_weight)[variant])
        eval_train_sanity.extend(examples_by_variant[variant][: min(9, len(examples_by_variant[variant]))])
    write_jsonl(out_dir / "eval_clean_dev.jsonl", eval_clean_dev)
    write_jsonl(out_dir / "eval_train_sanity.jsonl", eval_train_sanity)

    repair_path = package_dir / "seed_v1_splits/qwen_executable_seed_v1_final_repair.jsonl"
    if repair_path.is_file():
        repair_rows = load_jsonl(repair_path)
        repair_examples = []
        for row in repair_rows:
            try:
                repair_examples.extend(build_examples([row], package_dir, args.repeat_scale, args.min_answer_weight)["dagig"])
            except Exception:
                continue
        write_jsonl(out_dir / "eval_repair_if_available.jsonl", repair_examples)
    else:
        write_jsonl(out_dir / "eval_repair_if_available.jsonl", [])

    write_summary(out_dir / "sft_build_summary.json", train_rows, examples_by_variant)


if __name__ == "__main__":
    main()
