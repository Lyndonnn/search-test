from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


OOM_DOWNGRADES = [
    ("data.max_response_length", 768),
    ("train.group_size", 2),
    ("data.max_prompt_length", 2048),
    ("model.lora_r", 32),
    ("data.image_max_pixels", 512 * 512),
    ("train.per_device_batch_size", 1),
]


def load_config(path: str) -> dict[str, Any]:
    if yaml is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(path: str, cfg: dict[str, Any]) -> None:
    if yaml is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def apply_oom_downgrade(cfg: dict[str, Any]) -> dict[str, Any]:
    downgraded = deepcopy(cfg)
    for dotted_key, value in OOM_DOWNGRADES:
        _set_dotted(downgraded, dotted_key, value)
    return downgraded


def _set_dotted(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value

