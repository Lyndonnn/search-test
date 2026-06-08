from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any


def normalize_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def sample_id_from_extra_info(extra_info: dict[str, Any] | None) -> str:
    if not isinstance(extra_info, dict):
        return ""
    for key in ("question_id", "sample_id", "id", "index"):
        value = normalize_id(extra_info.get(key))
        if value:
            return value
    return ""


@lru_cache(maxsize=8)
def load_dagig_offline_edges(path: str) -> dict[str, dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return {}

    edges: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            edge = row.get("dependency_edge") if isinstance(row, dict) else None
            if not isinstance(edge, dict):
                continue
            sample_id = normalize_id(edge.get("sample_id") or row.get("sample_id"))
            if not sample_id:
                continue
            if not bool(row.get("selected_for_dependency_training", True)):
                continue
            edges[sample_id] = edge
    return edges


def dagig_edge_for_extra_info(extra_info: dict[str, Any] | None, path: str) -> dict[str, Any] | None:
    sample_id = sample_id_from_extra_info(extra_info)
    if not sample_id:
        return None
    return load_dagig_offline_edges(path).get(sample_id)


def edge_tool_type(edge: dict[str, Any], default: str = "image_search") -> str:
    return normalize_id(edge.get("step0_tool") or edge.get("tool_type") or default)


def edge_weight(edge: dict[str, Any], key: str = "constant") -> float:
    if key in {"", "none", "constant"}:
        return 1.0
    try:
        value = float(edge.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(value, 1.0))
