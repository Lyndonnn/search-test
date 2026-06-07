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

    # Stable aliases used by paper table scripts.
    val_score_keys = [k for k in metrics if k.startswith("val/") and k.endswith("/score")]
    val_reward_keys = [k for k in metrics if k.startswith("val/") and k.endswith("/reward")]
    text_keys = [k for k in metrics if k.startswith("val/") and k.endswith("/search_ratio_text")]
    image_keys = [k for k in metrics if k.startswith("val/") and k.endswith("/search_ratio_image")]
    mix_keys = [k for k in metrics if k.startswith("val/") and k.endswith("/search_ratio_mix")]
    fail_text_keys = [k for k in metrics if k.startswith("val/") and k.endswith("/search_fail_ratio_text")]
    fail_image_keys = [k for k in metrics if k.startswith("val/") and k.endswith("/search_fail_ratio_image")]

    if val_score_keys:
        metrics["val_score"] = metrics[val_score_keys[0]]
    if val_reward_keys:
        metrics["val_reward"] = metrics[val_reward_keys[0]]
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

