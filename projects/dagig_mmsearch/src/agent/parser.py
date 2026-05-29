from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ParsedAction:
    tool_type: str
    action_text: str
    arguments: dict[str, Any]
    valid: bool
    error: str = ""


def _extract_json_object(text: str) -> str | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def parse_action(text: str) -> ParsedAction:
    raw = text.strip()
    obj_text = _extract_json_object(raw)
    if obj_text:
        try:
            obj = json.loads(obj_text)
            tool = str(obj.get("tool") or obj.get("tool_type") or "").strip()
            action = obj.get("action", obj.get("query", obj.get("answer", "")))
            if tool:
                return ParsedAction(tool, str(action), obj, True)
        except Exception as exc:
            return _fallback_parse(raw, f"json_error={exc}")
    return _fallback_parse(raw, "no_json_object")


def _fallback_parse(text: str, error: str) -> ParsedAction:
    text_l = text.lower()
    text_match = re.search(r"<text_search>(.*?)</text_search>", text, re.DOTALL | re.IGNORECASE)
    if text_match:
        return ParsedAction("text_search", text_match.group(1).strip(), {"query": text_match.group(1).strip()}, True)
    if "<search><img></search>" in text_l or "<image_search>" in text_l:
        return ParsedAction("image_search", "current image", {"query_or_image": "current image"}, True)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        return ParsedAction("stop", answer_match.group(1).strip(), {"answer": answer_match.group(1).strip()}, True)
    return ParsedAction("stop", text.strip(), {"answer": text.strip()}, False, error=error)

