from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from reward.types import CounterfactualObservation, ToolStep


def _obs_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.split())
    if isinstance(value, dict):
        return sum(_obs_len(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_obs_len(v) for v in value)
    return len(str(value).split())


def _count_items(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict) and "results" in value and isinstance(value["results"], list):
        return len(value["results"])
    return 1


def _entity_tokens(text: str) -> set[str]:
    return {tok.lower() for tok in text.replace("_", " ").split() if len(tok) > 2}


class TypedCounterfactualPool:
    def __init__(self, max_per_type: int = 2048, seed: int = 42) -> None:
        self.max_per_type = max_per_type
        self.rng = random.Random(seed)
        self.by_type: dict[str, list[ToolStep]] = defaultdict(list)

    def add(self, step: ToolStep) -> None:
        bucket = self.by_type[step.tool_type]
        if not any(existing.step_id == step.step_id and existing.action_text == step.action_text for existing in bucket):
            bucket.append(step)
        if len(bucket) > self.max_per_type:
            del bucket[: len(bucket) - self.max_per_type]

    def add_many(self, steps: list[ToolStep]) -> None:
        for step in steps:
            self.add(step)

    def sample(self, tool_type: str, current_step: ToolStep, k: int) -> list[CounterfactualObservation]:
        candidates = [
            step
            for step in self.by_type.get(tool_type, [])
            if not (step.step_id == current_step.step_id and step.action_text == current_step.action_text)
        ]
        if not candidates:
            return [self._masked_counterfactual(tool_type, current_step, idx) for idx in range(k)]

        ranked = sorted(candidates, key=lambda step: self._candidate_key(tool_type, current_step, step), reverse=True)
        if len(ranked) < k:
            repeated = ranked + [self._step_from_mask(tool_type, current_step, idx) for idx in range(k - len(ranked))]
        else:
            repeated = ranked[:k]

        return [self._to_counterfactual(tool_type, current_step, candidate) for candidate in repeated[:k]]

    def _candidate_key(self, tool_type: str, current: ToolStep, candidate: ToolStep) -> tuple[float, float, float]:
        length_ratio = self._length_ratio(current, candidate)
        count_match = 1.0 if _count_items(current.raw_observation) == _count_items(candidate.raw_observation) else 0.0
        hard_negative = 1.0 if self._is_hard_negative(tool_type, current, candidate) else 0.0
        return (hard_negative, count_match, 1.0 - abs(1.0 - length_ratio))

    def _length_ratio(self, current: ToolStep, candidate: ToolStep) -> float:
        cur_len = max(1, _obs_len(current.raw_observation))
        cand_len = max(1, _obs_len(candidate.raw_observation))
        return cand_len / cur_len

    def _is_hard_negative(self, tool_type: str, current: ToolStep, candidate: ToolStep) -> bool:
        if current.metadata.get("answer") and candidate.metadata.get("answer"):
            if current.metadata.get("answer") != candidate.metadata.get("answer"):
                return True
        cur_entities = _entity_tokens(current.action_text + " " + current.evidence_summary)
        cand_entities = _entity_tokens(candidate.action_text + " " + candidate.evidence_summary)
        overlap = len(cur_entities & cand_entities)
        if tool_type in {"text_search", "ocr"}:
            return overlap > 0 and current.evidence_summary != candidate.evidence_summary
        if tool_type == "image_search":
            return overlap > 0 and current.metadata.get("semantic_id") != candidate.metadata.get("semantic_id")
        if tool_type == "select":
            return current.action_text != candidate.action_text
        if tool_type == "crop":
            return current.metadata.get("image_id") == candidate.metadata.get("image_id")
        return False

    def _to_counterfactual(
        self, tool_type: str, current: ToolStep, candidate: ToolStep | CounterfactualObservation
    ) -> CounterfactualObservation:
        if isinstance(candidate, CounterfactualObservation):
            return candidate
        length_ratio = self._length_ratio(current, candidate)
        metadata = {
            "cf_source": candidate.metadata.get("sample_id", "pool"),
            "cf_tool_type": tool_type,
            "cf_similarity": round(1.0 - abs(1.0 - length_ratio), 4),
            "cf_length_ratio": round(length_ratio, 4),
            "whether_hard_negative": self._is_hard_negative(tool_type, current, candidate),
        }
        return CounterfactualObservation(
            raw_observation=candidate.raw_observation,
            evidence_summary=candidate.evidence_summary,
            metadata=metadata,
        )

    def _masked_counterfactual(self, tool_type: str, current: ToolStep, idx: int) -> CounterfactualObservation:
        metadata = {
            "cf_source": f"masked_fallback_{idx}",
            "cf_tool_type": tool_type,
            "cf_similarity": 0.0,
            "cf_length_ratio": 1.0,
            "whether_hard_negative": False,
        }
        if tool_type == "stop":
            summary = "Low-information context: no reliable evidence available."
            raw = {"masked": True, "reason": "stop counterfactual"}
        elif tool_type == "crop":
            summary = "bbox=[0, 0, 1, 1] crop_path= visual=neighboring uninformative region"
            raw = {"bbox": [0, 0, 1, 1], "crop_path": "", "masked": True}
        elif tool_type == "ocr":
            summary = "OCR text: [masked similar-length text]"
            raw = {"text": "[masked similar-length text]"}
        elif tool_type == "image_search":
            summary = "1. image_id=masked caption=unrelated visual result score=0"
            raw = [{"image_id": "masked", "caption": "unrelated visual result", "score": 0.0}]
        else:
            summary = "1. masked result: unrelated search result with similar format"
            raw = [{"title": "masked result", "snippet": "unrelated search result with similar format", "score": 0.0}]
        return CounterfactualObservation(raw_observation=raw, evidence_summary=summary, metadata=metadata)

    def _step_from_mask(self, tool_type: str, current: ToolStep, idx: int) -> CounterfactualObservation:
        return self._masked_counterfactual(tool_type, current, idx)

