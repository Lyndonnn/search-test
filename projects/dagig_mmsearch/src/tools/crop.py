from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tools.base import ToolResult, summarize_observation
from utils.io import ensure_dir


class CropTool:
    tool_type = "crop"

    def __init__(self, output_dir: str = "data/cache/crops", max_summary_tokens: int = 96) -> None:
        self.output_dir = Path(output_dir)
        self.max_summary_tokens = max_summary_tokens

    def run(self, action_text: str, **kwargs: Any) -> ToolResult:
        image_path = kwargs.get("image_path", "")
        bbox = kwargs.get("bbox") or self._parse_bbox(action_text)
        crop_path = ""
        success = True
        if image_path and os.path.exists(image_path):
            try:
                from PIL import Image

                ensure_dir(self.output_dir)
                image = Image.open(image_path).convert("RGB")
                width, height = image.size
                x1, y1, x2, y2 = self._clamp_bbox(bbox, width, height)
                crop = image.crop((x1, y1, x2, y2))
                crop_path = str(self.output_dir / f"crop_{abs(hash((image_path, tuple(bbox))))}.png")
                crop.save(crop_path)
                bbox = [x1, y1, x2, y2]
            except Exception:
                success = False
        raw = {"bbox": bbox, "crop_path": crop_path, "visual_description": "cropped visual region"}
        return ToolResult(
            tool_type=self.tool_type,
            raw_observation=raw,
            evidence_summary=summarize_observation(self.tool_type, raw, self.max_summary_tokens),
            success=success,
            metadata={"image_path": image_path},
        )

    def _parse_bbox(self, text: str) -> list[int]:
        import re

        nums = [int(float(x)) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
        if len(nums) >= 4:
            return nums[:4]
        return [0, 0, 1, 1]

    def _clamp_bbox(self, bbox: list[int], width: int, height: int) -> list[int]:
        x1, y1, x2, y2 = bbox[:4]
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(x1 + 1, min(width, int(x2)))
        y2 = max(y1 + 1, min(height, int(y2)))
        return [x1, y1, x2, y2]

