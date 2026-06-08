#!/usr/bin/env python3
"""Extract final MMSearch-R1 validation metrics from a Hydra main_ppo.log."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from typing import Any


METRIC_RE = re.compile(r"(?:^| - )([A-Za-z0-9_./]+):([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="", help="main_ppo.log path. If omitted, use newest outputs/*/*/main_ppo.log.")
    parser.add_argument("--method", default="", help="Method name stored in the output row.")
    parser.add_argument("--output-csv", default="", help="Optional CSV path.")
    parser.add_argument("--output-json", default="", help="Optional JSON path.")
    return parser.parse_args()


def newest_log() -> str:
    candidates = glob.glob("outputs/*/*/main_ppo.log")
    if not candidates:
        raise FileNotFoundError("No Hydra log found under outputs/*/*/main_ppo.log")
    return max(candidates, key=os.path.getmtime)


def extract_metrics(path: str, method: str = "") -> dict[str, Any]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected a JSON object in {path}, got {type(loaded)}")
        metrics = {"method": method or str(loaded.get("experiment_name") or os.path.basename(os.path.dirname(path))), "source_log": path}
        metrics.update(loaded)
        metrics.update(load_last_train_row(path))
        return add_stable_aliases(metrics)

    final_line = ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "step:" in line and "val/" in line:
                final_line = line.strip()

    if not final_line:
        raise RuntimeError(f"No training metric line with validation metrics found in {path}")

    metrics: dict[str, Any] = {"method": method or os.path.basename(os.path.dirname(path)), "source_log": path}
    for key, value in METRIC_RE.findall(final_line):
        metrics[key] = float(value)

    return add_stable_aliases(metrics)


def load_last_train_row(path: str) -> dict[str, Any]:
    metrics_jsonl = os.path.join(os.path.dirname(path), "metrics.jsonl")
    if not os.path.isfile(metrics_jsonl):
        return {}
    last_train = None
    with open(metrics_jsonl, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("label") == "train":
                last_train = row
    if not last_train:
        return {}
    out = {}
    for key, value in last_train.items():
        if key in {"label", "method", "source_log"}:
            continue
        out[f"last_train/{key}"] = value
    return out


def add_stable_aliases(metrics: dict[str, Any]) -> dict[str, Any]:
    # Stable aliases used by paper table scripts.
    val_score_keys = [k for k in metrics if k.startswith("val/") and k.endswith("/score")]
    val_reward_keys = [k for k in metrics if k.startswith("val/") and k.endswith("/reward")]
    answer_score_keys = [k for k in metrics if k.startswith("val/") and k.endswith("/answer_score")]
    answer_reward_keys = [k for k in metrics if k.startswith("val/") and k.endswith("/answer_reward")]
    shaped_score_keys = [k for k in metrics if k.startswith("val/") and k.endswith("/shaped_score")]
    shaped_reward_keys = [k for k in metrics if k.startswith("val/") and k.endswith("/shaped_reward")]
    text_keys = [k for k in metrics if k.startswith("val/") and k.endswith("/search_ratio_text")]
    image_keys = [k for k in metrics if k.startswith("val/") and k.endswith("/search_ratio_image")]
    mix_keys = [k for k in metrics if k.startswith("val/") and k.endswith("/search_ratio_mix")]
    fail_text_keys = [k for k in metrics if k.startswith("val/") and k.endswith("/search_fail_ratio_text")]
    fail_image_keys = [k for k in metrics if k.startswith("val/") and k.endswith("/search_fail_ratio_image")]

    if val_score_keys:
        metrics["val_score"] = metrics[val_score_keys[0]]
    if val_reward_keys:
        metrics["val_reward"] = metrics[val_reward_keys[0]]
    if answer_score_keys:
        metrics["val_answer_score"] = metrics[answer_score_keys[0]]
    if answer_reward_keys:
        metrics["val_answer_reward"] = metrics[answer_reward_keys[0]]
    if shaped_score_keys:
        metrics["val_shaped_score"] = metrics[shaped_score_keys[0]]
    if shaped_reward_keys:
        metrics["val_shaped_reward"] = metrics[shaped_reward_keys[0]]
    if text_keys:
        metrics["val_search_ratio_text"] = metrics[text_keys[0]]
    if image_keys:
        metrics["val_search_ratio_image"] = metrics[image_keys[0]]
    if mix_keys:
        metrics["val_search_ratio_mix"] = metrics[mix_keys[0]]
    if fail_text_keys:
        metrics["val_search_fail_ratio_text"] = metrics[fail_text_keys[0]]
    if fail_image_keys:
        metrics["val_search_fail_ratio_image"] = metrics[fail_image_keys[0]]

    train_bonus_keys = [k for k in metrics if k.endswith("train/reward_diag_bonus_applied_rate")]
    train_edge_keys = [k for k in metrics if k.endswith("train/reward_diag_selected_edge_hit_rate")]
    train_effective_keys = [k for k in metrics if k.endswith("train/reward_diag_effective_search_rate")]
    train_invalid_keys = [k for k in metrics if k.endswith("train/reward_diag_invalid_action_rate")]
    train_raw_answer_keys = [k for k in metrics if k.endswith("train/reward_diag_raw_answer_reward")]
    train_before_keys = [k for k in metrics if k.endswith("train/reward_diag_final_reward_before_shaping")]
    train_after_keys = [k for k in metrics if k.endswith("train/reward_diag_final_reward_after_shaping")]
    if train_bonus_keys:
        metrics["last_train_bonus_applied_rate"] = metrics[train_bonus_keys[0]]
    if train_edge_keys:
        metrics["last_train_selected_edge_hit_rate"] = metrics[train_edge_keys[0]]
    if train_effective_keys:
        metrics["last_train_effective_search_rate"] = metrics[train_effective_keys[0]]
    if train_invalid_keys:
        metrics["last_train_invalid_action_rate"] = metrics[train_invalid_keys[0]]
    if train_raw_answer_keys:
        metrics["last_train_raw_answer_reward"] = metrics[train_raw_answer_keys[0]]
    if train_before_keys:
        metrics["last_train_final_reward_before_shaping"] = metrics[train_before_keys[0]]
    if train_after_keys:
        metrics["last_train_final_reward_after_shaping"] = metrics[train_after_keys[0]]

    return metrics


def write_csv(path: str, row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    path = args.input or newest_log()
    metrics = extract_metrics(path, method=args.method)
    for key, value in metrics.items():
        print(f"{key}={value}")
    if args.output_csv:
        write_csv(args.output_csv, metrics)
        print(f"wrote_csv={args.output_csv}")
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"wrote_json={args.output_json}")


if __name__ == "__main__":
    main()
