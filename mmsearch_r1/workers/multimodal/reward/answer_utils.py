"""Answer field normalization helpers for MMSearch-R1 rewards."""

from __future__ import annotations

import json
from typing import Any


def normalize_answer_list(value: Any) -> list[str]:
    """Normalize parquet-loaded answer fields into a list of strings.

    PyArrow/Pandas can materialize nested list columns as numpy.ndarray, while
    the original MMSearch-R1 parquet stores candidate_answers as a JSON string.
    The reward path should accept both representations.
    """

    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return normalize_answer_list(parsed)
        return [text]
    if hasattr(value, "tolist"):
        return normalize_answer_list(value.tolist())
    if isinstance(value, (list, tuple, set)):
        answers: list[str] = []
        for item in value:
            answers.extend(normalize_answer_list(item))
        return answers
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore").strip()
        return [text] if text else []
    return [str(value).strip()] if str(value).strip() else []
