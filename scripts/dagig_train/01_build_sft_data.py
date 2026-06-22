#!/usr/bin/env python3
"""Build segment-weighted SFT datasets from the clean Pix2Fact-DAGIG package."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_PACKAGE_NAME = "pix2fact_dagig_1k_gpt54_teacher_clean_package"
DEFAULT_MAIN_FILE = "data/pix2fact_dagig_train_AB_clean_split.jsonl"
VARIANTS = ["uniform_sft", "outcome_only_sft", "local_ig_sft", "dagig_sft", "dagig_action_only_sft"]
SEGMENTS = ["observe", "search_decision", "search", "evidence", "answer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package_dir", default=f"data/{DEFAULT_PACKAGE_NAME}")
    parser.add_argument("--out_dir", default="data/dagig_sft")
    parser.add_argument("--main_file", default=DEFAULT_MAIN_FILE)
    parser.add_argument("--min_process_weight", type=float, default=0.05)
    parser.add_argument("--include_test", action="store_true", help="Also write held-out test JSONL for evaluation.")
    parser.add_argument("--debug_examples", type=int, default=10)
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


def teacher(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("gpt54_teacher")
    if not isinstance(value, dict):
        raise ValueError(f"Missing gpt54_teacher for sample_id={row.get('sample_id')}")
    return value


def action_rewards(row: dict[str, Any]) -> dict[str, float]:
    rewards = teacher(row).get("action_rewards")
    if not isinstance(rewards, dict):
        raise ValueError(f"Missing action_rewards for sample_id={row.get('sample_id')}")

    def get(key: str) -> float:
        try:
            return max(0.0, min(1.0, float(rewards.get(key, 0.0))))
        except (TypeError, ValueError):
            return 0.0

    return {
        "observe_crop": get("observe_crop"),
        "search_query": get("search_query"),
        "evidence_selection": get("evidence_selection"),
        "answer": get("answer"),
    }


def training_weight(row: dict[str, Any]) -> float:
    try:
        return float(teacher(row).get("training_weight", 1.0))
    except (TypeError, ValueError):
        return 1.0


def resolve_image_path(package_dir: Path, row: dict[str, Any], key: str) -> str:
    package_paths = row.get("package_image_paths")
    if isinstance(package_paths, dict) and package_paths.get(key):
        path = Path(str(package_paths[key]))
        resolved = path if path.is_absolute() else package_dir / path
    else:
        image_paths = row.get("image_paths")
        if not isinstance(image_paths, dict) or not image_paths.get(key):
            raise FileNotFoundError(f"Missing image path key={key} for sample_id={row.get('sample_id')}")
        raw = Path(str(image_paths[key]))
        resolved = raw if raw.is_file() else package_dir / "images" / key / raw.name
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing image file for sample_id={row.get('sample_id')} key={key}: {resolved}")
    return str(resolved.resolve())


def evidence_text(row: dict[str, Any]) -> str:
    t = teacher(row)
    quote = str(t.get("supporting_evidence_quote", "")).strip()
    if quote:
        return quote
    evidences = row.get("evidences")
    if isinstance(evidences, list):
        for item in evidences:
            if isinstance(item, dict) and item.get("answer_supported"):
                text = str(item.get("text", "")).strip()
                if text:
                    return text
        for item in evidences:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                if text:
                    return text
    return ""


def bool_text(value: Any) -> str:
    if isinstance(value, str):
        return "true" if value.lower() in {"1", "true", "yes", "y"} else "false"
    return "true" if bool(value) else "false"


def target_segments(row: dict[str, Any]) -> dict[str, str]:
    t = teacher(row)
    return {
        "observe": str(t.get("local_observation", "")).strip(),
        "search_decision": bool_text(t.get("search_needed", True)),
        "search": str(t.get("repaired_search_query") or row.get("hf_search_query") or "").strip(),
        "evidence": evidence_text(row),
        "answer": str(row.get("answer", "")).strip(),
    }


def format_target(segments: dict[str, str]) -> str:
    return "\n".join(f"<{name}>\n{segments[name]}\n</{name}>" for name in SEGMENTS)


def prompt_text(row: dict[str, Any], has_oracle_crop: bool = True) -> str:
    image_text = "the full image and the crop" if has_oracle_crop else "the provided region-normalized image"
    return (
        f"You are a multimodal search agent. Use {image_text} to produce a grounded "
        "search trajectory. Return exactly these XML-like sections: <observe>, <search_decision>, "
        "<search>, <evidence>, <answer>.\n\n"
        f"Question: {str(row.get('question', '')).strip()}"
    )


def variant_segment_weights(row: dict[str, Any], min_process_weight: float) -> dict[str, dict[str, float]]:
    w = training_weight(row)
    r = action_rewards(row)
    search_decision_reward = r["search_query"] if teacher(row).get("search_needed", True) else r["answer"]
    return {
        "uniform_sft": {
            "observe": w,
            "search_decision": w,
            "search": w,
            "evidence": w,
            "answer": w,
        },
        "outcome_only_sft": {
            "observe": w * min_process_weight,
            "search_decision": w * min_process_weight,
            "search": w * min_process_weight,
            "evidence": w * min_process_weight,
            "answer": w * r["answer"],
        },
        "local_ig_sft": {
            "observe": w * r["observe_crop"],
            "search_decision": w * min_process_weight,
            "search": w * min_process_weight,
            "evidence": w * r["evidence_selection"],
            "answer": w * r["answer"],
        },
        "dagig_sft": {
            "observe": w * r["observe_crop"],
            "search_decision": w * search_decision_reward,
            "search": w * r["search_query"],
            "evidence": w * r["evidence_selection"],
            "answer": w * r["answer"],
        },
        "dagig_action_only_sft": {
            "observe": w * r["observe_crop"],
            "search_decision": w * search_decision_reward,
            "search": w * r["search_query"],
            "evidence": w * min_process_weight,
            "answer": w * r["answer"],
        },
    }


def make_example(package_dir: Path, row: dict[str, Any], variant: str, min_process_weight: float) -> dict[str, Any]:
    segments = target_segments(row)
    if not segments["observe"] or not segments["search"] or not segments["answer"]:
        raise ValueError(f"Empty required target segment for sample_id={row.get('sample_id')}")
    weights = variant_segment_weights(row, min_process_weight)[variant]
    target = format_target(segments)
    t = teacher(row)
    full_image = resolve_image_path(package_dir, row, "full_model_input")
    try:
        crop_image = resolve_image_path(package_dir, row, "crop_model_input")
    except FileNotFoundError:
        crop_image = ""
    images = [full_image] + ([crop_image] if crop_image else [])
    prompt = prompt_text(row, has_oracle_crop=bool(crop_image))
    return {
        "sample_id": row.get("sample_id"),
        "variant": variant,
        "split": row.get("split"),
        "task_type": "oracle_crop_chain" if crop_image else "region_normalized_chain",
        "input_mode": "full_plus_oracle_crop" if crop_image else "region_normalized_image_only",
        "prompt": prompt,
        "target": target,
        "target_segments": segments,
        "segment_weights": {name: float(weights[name]) for name in SEGMENTS},
        "loss_weight": sum(float(weights[name]) for name in SEGMENTS) / len(SEGMENTS),
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "images": images,
        "full_image_path": full_image,
        "crop_image_path": crop_image,
        "question": row.get("question"),
        "answer": row.get("answer"),
        "bbox_xyxy": row.get("bbox_xyxy"),
        "image_width": row.get("image_width"),
        "image_height": row.get("image_height"),
        "tier": t.get("tier"),
        "training_weight": training_weight(row),
        "visual_anchor": t.get("visual_anchor"),
        "visual_anchor_type": t.get("visual_anchor_type"),
        "search_needed": bool(t.get("search_needed", True)),
        "query_specificity": t.get("query_specificity"),
        "evidence_support": t.get("evidence_support"),
        "delayed_credit_strength": t.get("delayed_credit_strength"),
        "action_rewards": action_rewards(row),
        "reward_variants": row.get("reward_variants", {}),
        "evidences": row.get("evidences", []),
        "group_key": row.get("group_key"),
    }


def split_examples(rows: list[dict[str, Any]], package_dir: Path, min_process_weight: float) -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = {variant: {"train": [], "dev": [], "test": []} for variant in VARIANTS}
    for row in rows:
        split = str(row.get("split", "train"))
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"Unsupported split={split!r} for sample_id={row.get('sample_id')}")
        row_tier = str(teacher(row).get("tier", ""))
        if row_tier not in {"A", "B"}:
            raise ValueError(f"Non A/B row found in main file: sample_id={row.get('sample_id')} tier={row_tier}")
        if not bool(teacher(row).get("keep_for_training", False)):
            raise ValueError(f"keep_for_training=false in main file: sample_id={row.get('sample_id')}")
        for variant in VARIANTS:
            out[variant][split].append(make_example(package_dir, row, variant, min_process_weight))
    return out


def write_debug(path: Path, rows: list[dict[str, Any]], examples_by_variant: dict[str, dict[str, list[dict[str, Any]]]], n: int) -> None:
    debug = {
        "source_n": len(rows),
        "source_split_counts": dict(Counter(str(r.get("split")) for r in rows)),
        "source_tier_counts": dict(Counter(str(teacher(r).get("tier")) for r in rows)),
        "variants": {},
        "examples": [],
    }
    for variant, splits in examples_by_variant.items():
        debug["variants"][variant] = {}
        for split, examples in splits.items():
            debug["variants"][variant][split] = {
                "n": len(examples),
                "mean_loss_weight": sum(float(ex["loss_weight"]) for ex in examples) / max(1, len(examples)),
                "segment_weight_means": {
                    seg: sum(float(ex["segment_weights"][seg]) for ex in examples) / max(1, len(examples)) for seg in SEGMENTS
                },
            }
    for ex in examples_by_variant["dagig_sft"]["train"][:n]:
        debug["examples"].append(
            {
                "sample_id": ex["sample_id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "visual_anchor": ex["visual_anchor"],
                "target_segments": ex["target_segments"],
                "action_rewards": ex["action_rewards"],
                "segment_weights": ex["segment_weights"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    package_dir = Path(args.package_dir).expanduser().resolve()
    main_path = Path(args.main_file)
    if not main_path.is_absolute():
        main_path = package_dir / main_path
    if not main_path.is_file():
        raise FileNotFoundError(f"Training file missing: {main_path}")

    rows = load_jsonl(main_path)
    examples_by_variant = split_examples(rows, package_dir, args.min_process_weight)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dev: list[dict[str, Any]] = []
    all_test: list[dict[str, Any]] = []
    for variant, splits in examples_by_variant.items():
        write_jsonl(out_dir / f"{variant}_train.jsonl", splits["train"])
        write_jsonl(out_dir / f"{variant}_dev.jsonl", splits["dev"])
        if args.include_test:
            write_jsonl(out_dir / f"{variant}_test.jsonl", splits["test"])
        all_dev.extend(splits["dev"])
        all_test.extend(splits["test"])

    write_jsonl(out_dir / "eval_all_variants_dev.jsonl", all_dev)
    if args.include_test:
        write_jsonl(out_dir / "eval_all_variants_test.jsonl", all_test)
    write_jsonl(out_dir / "eval_train_sanity.jsonl", examples_by_variant["dagig_sft"]["train"][: min(20, len(examples_by_variant["dagig_sft"]["train"]))])
    write_debug(out_dir / "sft_build_summary.json", rows, examples_by_variant, args.debug_examples)


if __name__ == "__main__":
    main()
