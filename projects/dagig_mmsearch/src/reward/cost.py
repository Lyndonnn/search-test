from __future__ import annotations

from reward.types import ToolStep


DEFAULT_TOOL_COSTS = {
    "text_search": 1.0,
    "image_search": 1.5,
    "crop": 0.5,
    "ocr": 0.8,
    "select": 0.2,
    "stop": 0.0,
}


class ToolCostModel:
    def __init__(self, costs: dict[str, float] | None = None) -> None:
        self.costs = dict(DEFAULT_TOOL_COSTS)
        if costs:
            self.costs.update(costs)

    def score(self, step: ToolStep) -> float:
        return float(self.costs.get(step.tool_type, 1.0))

