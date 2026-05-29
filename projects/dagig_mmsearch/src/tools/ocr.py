from __future__ import annotations

from typing import Any

from tools.base import ToolResult, summarize_observation


class OCRTool:
    tool_type = "ocr"

    def __init__(self, max_summary_tokens: int = 96) -> None:
        self.max_summary_tokens = max_summary_tokens

    def run(self, action_text: str, **kwargs: Any) -> ToolResult:
        image_path = kwargs.get("image_path", action_text)
        text = kwargs.get("text", "")
        success = True
        if not text and image_path:
            try:
                from PIL import Image
                import pytesseract

                text = pytesseract.image_to_string(Image.open(image_path))
            except Exception:
                text = kwargs.get("fallback_text", "")
                success = bool(text)
        raw = {"text": text, "image_path": image_path}
        return ToolResult(
            tool_type=self.tool_type,
            raw_observation=raw,
            evidence_summary=summarize_observation(self.tool_type, raw, self.max_summary_tokens),
            success=success,
            metadata={"image_path": image_path},
        )

