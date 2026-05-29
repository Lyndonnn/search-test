from __future__ import annotations

import difflib
import re
from typing import Any

from reward.types import ToolStep, Trajectory


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


class GateRewardScorer:
    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        tool_free_probe_n: int = 3,
        consistency_threshold: float = 0.8,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tool_free_probe_n = tool_free_probe_n
        self.consistency_threshold = consistency_threshold

    def score(self, trajectory: Trajectory, step: ToolStep) -> float:
        answers = self._probe_answers(trajectory)
        consistency = self._consistency(answers)
        consensus = self._consensus(answers)
        action_match = self._similarity(step.action_text, consensus)
        is_stop = step.tool_type == "stop"
        if consistency >= self.consistency_threshold and is_stop and action_match >= self.consistency_threshold:
            return 1.0
        if consistency >= self.consistency_threshold and is_stop:
            return -0.5
        if consistency >= self.consistency_threshold and not is_stop:
            return -0.25
        if consistency < self.consistency_threshold and not is_stop:
            return 0.5
        return -0.5

    def _probe_answers(self, trajectory: Trajectory) -> list[str]:
        if "tool_free_answers" in trajectory.metadata:
            return [str(item) for item in trajectory.metadata["tool_free_answers"][: self.tool_free_probe_n]]
        if self.model is not None and hasattr(self.model, "generate_text"):
            return [str(self.model.generate_text(trajectory.question)) for _ in range(self.tool_free_probe_n)]
        if trajectory.final_correct:
            answer = trajectory.gold_answers[0] if trajectory.gold_answers else trajectory.final_answer
            return [answer for _ in range(self.tool_free_probe_n)]
        return [trajectory.final_answer, "unknown", trajectory.question[:16]]

    def _consistency(self, answers: list[str]) -> float:
        norm = [normalize_text(answer) for answer in answers if answer]
        if len(norm) <= 1:
            return 1.0
        scores = []
        for i in range(len(norm)):
            for j in range(i + 1, len(norm)):
                scores.append(difflib.SequenceMatcher(None, norm[i], norm[j]).ratio())
        return sum(scores) / max(1, len(scores))

    def _consensus(self, answers: list[str]) -> str:
        norm = [normalize_text(answer) for answer in answers if answer]
        if not norm:
            return ""
        return max(set(norm), key=norm.count)

    def _similarity(self, a: str, b: str) -> float:
        a_norm = normalize_text(a)
        b_norm = normalize_text(b)
        if not a_norm or not b_norm:
            return 0.0
        if a_norm == b_norm:
            return 1.0
        if a_norm in b_norm or b_norm in a_norm:
            return 0.9
        return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
