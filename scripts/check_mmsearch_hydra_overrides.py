#!/usr/bin/env python3
"""Validate MMSearch-R1 bash overrides against the Hydra base config.

This catches failures like:
  Could not override 'actor_rollout_ref.model.attn_implementation'
  Key 'attn_implementation' is not in struct

It is intentionally lightweight and does not import Hydra, Ray, or torch.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SCRIPTS = [
    "mmsearch_r1/scripts/run_mmsearch_r1_val_only_a100_debug.sh",
    "mmsearch_r1/scripts/run_mmsearch_r1_grpo_a100_debug.sh",
    "mmsearch_r1/scripts/run_mmsearch_r1_grpo.sh",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="mmsearch_r1/trainer/multimodal/config/ppo_trainer.yaml",
        help="Hydra base config to validate against.",
    )
    parser.add_argument(
        "scripts",
        nargs="*",
        default=DEFAULT_SCRIPTS,
        help="Bash scripts containing dotted Hydra overrides.",
    )
    return parser.parse_args()


def dotted_key_exists(config: dict[str, Any], dotted_key: str) -> bool:
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return False
        cursor = cursor[part]
    return True


def iter_override_keys(script_path: Path) -> list[str]:
    pattern = re.compile(r"^([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+)=")
    keys: list[str] = []
    for raw_line in script_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if match:
            keys.append(match.group(1))
    return keys


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    has_missing = False

    for script in args.scripts:
        script_path = Path(script)
        keys = iter_override_keys(script_path)
        missing = [key for key in keys if not dotted_key_exists(config, key)]
        print(f"{script}: {len(keys)} override keys, {len(missing)} missing")
        for key in missing:
            print(f"  MISSING {key}")
        has_missing = has_missing or bool(missing)

    if has_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
