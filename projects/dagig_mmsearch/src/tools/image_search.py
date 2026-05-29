from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.base import ToolResult, summarize_observation
from utils.io import read_jsonl


class ImageSearchTool:
    tool_type = "image_search"

    def __init__(self, index_path: str = "data/indexes/image_corpus.jsonl", topk: int = 5, max_summary_tokens: int = 96):
        self.index_path = Path(index_path)
        self.topk = topk
        self.max_summary_tokens = max_summary_tokens
        self._index = read_jsonl(self.index_path)

    def run(self, action_text: str, **kwargs: Any) -> ToolResult:
        topk = int(kwargs.get("topk", self.topk))
        results = self._search(action_text, topk)
        return ToolResult(
            tool_type=self.tool_type,
            raw_observation=results,
            evidence_summary=summarize_observation(self.tool_type, results, self.max_summary_tokens),
            metadata={"query_or_image": action_text, "topk": topk, "backend": "local_jsonl"},
        )

    def _search(self, query: str, topk: int) -> list[dict[str, Any]]:
        query_tokens = set(query.lower().split())
        rows = self._index or self._fallback_index()
        scored = []
        for row in rows:
            text = f"{row.get('title', '')} {row.get('caption', '')} {row.get('image_id', '')}".lower()
            score = sum(token in text for token in query_tokens)
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for rank, (score, row) in enumerate(scored[:topk], start=1):
            result = dict(row)
            result.setdefault("image_id", f"image_{rank}")
            result.setdefault("caption", result.get("title", "visual search result"))
            result["score"] = float(score)
            results.append(result)
        return results

    def _fallback_index(self) -> list[dict[str, str]]:
        return [
            {
                "image_id": "img_eiffel",
                "title": "Eiffel Tower",
                "caption": "Iron lattice tower and Paris landmark.",
                "answer": "Paris",
            },
            {
                "image_id": "img_louvre",
                "title": "Louvre Museum",
                "caption": "Museum entrance and glass pyramid in Paris.",
                "answer": "Louvre Museum",
            },
            {
                "image_id": "img_bridge",
                "title": "Golden Gate Bridge",
                "caption": "Red suspension bridge in San Francisco.",
                "answer": "San Francisco",
            },
        ]

