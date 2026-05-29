from __future__ import annotations

from statistics import mean
from typing import Any

from reward.counterfactual import build_context, sample_counterfactuals
from reward.typed_pool import TypedCounterfactualPool
from reward.types import ToolStep, Trajectory
from utils.model_logprob import FrozenLogProbScorer


class LocalIGScorer:
    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        cf_pool: TypedCounterfactualPool | None = None,
        cf_samples: int = 4,
        dead_zone: float = 0.02,
        negative_scale: float = 0.25,
        length_norm: bool = True,
    ) -> None:
        self.scorer = FrozenLogProbScorer(model=model, tokenizer=tokenizer, length_norm=length_norm)
        self.cf_pool = cf_pool or TypedCounterfactualPool()
        self.cf_samples = cf_samples
        self.dead_zone = dead_zone
        self.negative_scale = negative_scale
        self.length_norm = length_norm
        self.last_diagnostics: dict[str, Any] = {}

    def score_step(self, trajectory: Trajectory, step: ToolStep) -> float:
        target = trajectory.gold_answers[0] if trajectory.gold_answers else trajectory.final_answer
        real_context = build_context(trajectory, before_step_id=step.step_id + 1)
        logp_real = self.scorer.score(real_context, target)
        cfs = sample_counterfactuals(self.cf_pool, step, self.cf_samples)
        cf_logps = [
            self.scorer.score(
                build_context(
                    trajectory,
                    replace_step_id=step.step_id,
                    replacement=cf,
                    before_step_id=step.step_id + 1,
                ),
                target,
            )
            for cf in cfs
        ]
        logp_cf = mean(cf_logps) if cf_logps else logp_real
        value = self._stabilize(logp_real - logp_cf)
        self.last_diagnostics = {
            "step_id": step.step_id,
            "tool_type": step.tool_type,
            "logp_real": logp_real,
            "logp_cf": logp_cf,
            "counterfactual_debug": [cf.metadata for cf in cfs],
        }
        return value

    def score_batch(self, trajectory: Trajectory, steps: list[ToolStep]) -> dict[int, float]:
        return {step.step_id: self.score_step(trajectory, step) for step in steps}

    def _stabilize(self, value: float) -> float:
        if abs(value) < self.dead_zone:
            value = 0.0
        if value < 0:
            value *= self.negative_scale
        return max(-2.0, min(2.0, float(value)))

