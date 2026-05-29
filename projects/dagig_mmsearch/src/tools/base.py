from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolResult:
    tool_type: str
    raw_observation: Any
    evidence_summary: str
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    tool_type: str

    def run(self, action_text: str, **kwargs: Any) -> ToolResult:
        ...


def truncate_tokens(text: str, max_tokens: int = 96) -> str:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


def _stringify_observation(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return " ".join(f"{k}: {_stringify_observation(v)}" for k, v in raw.items())
    if isinstance(raw, (list, tuple)):
        return " ".join(_stringify_observation(item) for item in raw)
    return str(raw)


def summarize_observation(tool_type: str, raw: Any, max_tokens: int = 96) -> str:
    if tool_type == "text_search":
        items = raw if isinstance(raw, list) else [{"title": "result", "snippet": _stringify_observation(raw)}]
        summaries = []
        for idx, item in enumerate(items[:5], start=1):
            title = str(item.get("title", f"result {idx}")) if isinstance(item, dict) else f"result {idx}"
            snippet = str(item.get("snippet", item.get("text", ""))) if isinstance(item, dict) else str(item)
            summaries.append(f"{idx}. {title}: {truncate_tokens(snippet, 80)}")
        return truncate_tokens(" ".join(summaries), max_tokens)

    if tool_type == "image_search":
        items = raw if isinstance(raw, list) else [{"image_id": "image_0", "caption": _stringify_observation(raw)}]
        summaries = []
        for idx, item in enumerate(items[:5], start=1):
            if isinstance(item, dict):
                image_id = item.get("image_id", item.get("id", f"image_{idx}"))
                caption = item.get("caption", item.get("title", ""))
                score = item.get("score", "")
                summaries.append(f"{idx}. image_id={image_id} caption={caption} score={score}")
            else:
                summaries.append(f"{idx}. image_id=image_{idx} caption={item}")
        return truncate_tokens(" ".join(summaries), max_tokens)

    if tool_type == "ocr":
        text = raw.get("text", "") if isinstance(raw, dict) else _stringify_observation(raw)
        return truncate_tokens(f"OCR text: {text}", max_tokens)

    if tool_type == "crop":
        if isinstance(raw, dict):
            bbox = raw.get("bbox", "")
            crop_path = raw.get("crop_path", "")
            desc = raw.get("visual_description", "cropped visual region")
            return truncate_tokens(f"bbox={bbox} crop_path={crop_path} visual={desc}", max_tokens)
        return truncate_tokens(_stringify_observation(raw), max_tokens)

    if tool_type == "select":
        if isinstance(raw, dict):
            return truncate_tokens(
                f"selected_item_id={raw.get('selected_item_id', '')} score={raw.get('score', '')}", max_tokens
            )
        return truncate_tokens(_stringify_observation(raw), max_tokens)

    if tool_type == "stop":
        return truncate_tokens(_stringify_observation(raw) or "No new observation.", max_tokens)

    cleaned = re.sub(r"\s+", " ", _stringify_observation(raw)).strip()
    return truncate_tokens(cleaned, max_tokens)

