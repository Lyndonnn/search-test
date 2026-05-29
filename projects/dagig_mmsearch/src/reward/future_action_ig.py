from __future__ import annotations

from statistics import mean
from typing import Any

from reward.counterfactual import build_context, sample_counterfactuals
from reward.typed_pool import TypedCounterfactualPool
from reward.types import ToolStep, Trajectory
from utils.model_logprob import FrozenLogProbScorer


class FutureActionIGScorer:
    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        cf_pool: TypedCounterfactualPool | None = None,
        cf_samples: int = 4,
        dead_zone: float = 0.02,
        positive_only: bool = True,
    ) -> None:
        self.scorer = FrozenLogProbScorer(model=model, tokenizer=tokenizer, length_norm=True)
        self.cf_pool = cf_pool or TypedCounterfactualPool()
        self.cf_samples = cf_samples
        self.dead_zone = dead_zone
        self.positive_only = positive_only
        self.last_diagnostics: dict[str, Any] = {}

    def score_edge(self, trajectory: Trajectory, source_step: ToolStep, target_step: ToolStep) -> float:
        target_action = target_step.action_text or target_step.tool_type
        real_context = build_context(trajectory, before_step_id=target_step.step_id)
        logp_real_action = self.scorer.score(real_context, target_action)
        cfs = sample_counterfactuals(self.cf_pool, source_step, self.cf_samples)
        cf_logps = [
            self.scorer.score(
                build_context(
                    trajectory,
                    replace_step_id=source_step.step_id,
                    replacement=cf,
                    before_step_id=target_step.step_id,
                ),
                target_action,
            )
            for cf in cfs
        ]
        logp_cf_action = mean(cf_logps) if cf_logps else logp_real_action
        value = logp_real_action - logp_cf_action
        if abs(value) < self.dead_zone:
            value = 0.0
        if self.positive_only:
            value = max(0.0, value)
        value = max(-2.0, min(2.0, float(value)))
        self.last_diagnostics = {
            "edge": [source_step.step_id, target_step.step_id],
            "source_tool_type": source_step.tool_type,
            "target_tool_type": target_step.tool_type,
            "target_action": target_action,
            "logp_real_action": logp_real_action,
            "logp_cf_action": logp_cf_action,
            "counterfactual_debug": [cf.metadata for cf in cfs],
        }
        return value

