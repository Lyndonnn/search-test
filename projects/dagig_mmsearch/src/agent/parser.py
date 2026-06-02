from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


ALLOWED_TOOLS = {"text_search", "image_search", "crop", "ocr", "select", "stop"}
PLACEHOLDER_ACTIONS = {"", "query", "answer", "final answer", "visual query", "current image"}


@dataclass
class ParsedAction:
    tool_type: str
    action_text: str
    arguments: dict[str, Any]
    valid: bool
    error: str = ""


def _extract_json_object(text: str) -> str | None:
    objects = _extract_json_objects(text)
    return objects[0] if objects else None


def _extract_json_objects(text: str) -> list[str]:
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    balanced = _balanced_json_objects(text)
    objects: list[str] = []
    seen = set()
    for item in [*fenced, *balanced]:
        key = item.strip()
        if key and key not in seen:
            objects.append(key)
            seen.add(key)
    return objects


def _balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    start = None
    depth = 0
    in_string = False
    escaped = False
    for idx, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : idx + 1])
                start = None
    return objects


def parse_action(text: str) -> ParsedAction:
    raw = text.strip()
    parsed_candidates: list[ParsedAction] = []
    json_errors: list[str] = []
    for obj_text in _extract_json_objects(raw):
        try:
            obj = json.loads(obj_text)
            parsed = _parse_json_action(obj)
            if parsed is not None:
                parsed_candidates.append(parsed)
        except Exception as exc:
            json_errors.append(str(exc))
    if parsed_candidates:
        return max(parsed_candidates, key=_candidate_score)
    error = f"json_error={json_errors[0]}" if json_errors else "no_json_object"
    return _fallback_parse(raw, error)


def _parse_json_action(obj: Any) -> ParsedAction | None:
    if not isinstance(obj, dict):
        return None
    tool = str(obj.get("tool") or obj.get("tool_type") or "").strip()
    if not tool:
        return None
    tool = tool.lower()
    action = _action_text_for_tool(tool, obj)
    valid = tool in ALLOWED_TOOLS
    error = "" if valid else f"unknown_tool={tool}"
    return ParsedAction(tool, action, obj, valid, error=error)


def _action_text_for_tool(tool: str, obj: dict[str, Any]) -> str:
    if tool in {"text_search", "image_search"}:
        return _first_specific_value(obj, ["query", "action", "image_query", "caption", "text"]) or str(
            obj.get("action", obj.get("query", ""))
        )
    if tool == "stop":
        return _first_specific_value(obj, ["answer", "final_answer", "action"]) or str(obj.get("action", obj.get("answer", "")))
    if tool == "select":
        return str(obj.get("index", obj.get("selected_index", obj.get("action", ""))))
    if tool == "crop":
        return str(obj.get("bbox", obj.get("region", obj.get("action", ""))))
    if tool == "ocr":
        return str(obj.get("region", obj.get("bbox", obj.get("action", ""))))
    return str(obj.get("action", obj.get("query", obj.get("answer", ""))))


def _first_specific_value(obj: dict[str, Any], keys: list[str]) -> str:
    fallback = ""
    for key in keys:
        value = obj.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not fallback and text:
            fallback = text
        if text.lower() not in PLACEHOLDER_ACTIONS:
            return text
    return fallback


def _candidate_score(parsed: ParsedAction) -> tuple[int, int, int]:
    valid_tool = int(parsed.tool_type in ALLOWED_TOOLS)
    specific_action = int(parsed.action_text.strip().lower() not in PLACEHOLDER_ACTIONS)
    is_stop = int(parsed.tool_type == "stop")
    return valid_tool, specific_action, -is_stop


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
