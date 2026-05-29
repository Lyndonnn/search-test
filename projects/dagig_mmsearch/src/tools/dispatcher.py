from __future__ import annotations

from typing import Any

from tools.base import ToolResult, summarize_observation
from tools.crop import CropTool
from tools.image_search import ImageSearchTool
from tools.ocr import OCRTool
from tools.text_search import TextSearchTool


class ToolDispatcher:
    def __init__(self, search_topk: int = 5, max_summary_tokens: int = 96) -> None:
        self.tools = {
            "text_search": TextSearchTool(topk=search_topk, max_summary_tokens=max_summary_tokens),
            "image_search": ImageSearchTool(topk=search_topk, max_summary_tokens=max_summary_tokens),
            "crop": CropTool(max_summary_tokens=max_summary_tokens),
            "ocr": OCRTool(max_summary_tokens=max_summary_tokens),
        }
        self.max_summary_tokens = max_summary_tokens

    def run(self, tool_type: str, action_text: str, **kwargs: Any) -> ToolResult:
        if tool_type == "stop":
            raw = {"final": action_text}
            return ToolResult(
                tool_type="stop",
                raw_observation=raw,
                evidence_summary=summarize_observation("stop", raw, self.max_summary_tokens),
                metadata={"action": action_text},
            )
        if tool_type == "select":
            raw = {
                "selected_item_id": kwargs.get("selected_item_id", action_text),
                "score": kwargs.get("score", 0.0),
            }
            return ToolResult(
                tool_type="select",
                raw_observation=raw,
                evidence_summary=summarize_observation("select", raw, self.max_summary_tokens),
                metadata={},
            )
        if tool_type not in self.tools:
            raw = {"error": f"unknown tool {tool_type}", "action": action_text}
            return ToolResult(tool_type=tool_type, raw_observation=raw, evidence_summary=str(raw), success=False)
        return self.tools[tool_type].run(action_text, **kwargs)

