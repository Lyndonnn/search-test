from __future__ import annotations

from dataclasses import asdict
from typing import Any

from reward.cost import ToolCostModel
from reward.future_action_ig import FutureActionIGScorer
from reward.gate import GateRewardScorer
from reward.local_ig import LocalIGScorer
from reward.propagation import dagig_lite_returns, inject_span_rewards
from reward.typed_pool import TypedCounterfactualPool
from reward.types import DAGIGOutput, RewardBatch, StepReward, Trajectory


class DAGIGLiteReward:
    def __init__(
        self,
        local_ig_scorer: LocalIGScorer | None = None,
        future_action_ig_scorer: FutureActionIGScorer | None = None,
        gate_scorer: GateRewardScorer | None = None,
        cost_model: ToolCostModel | None = None,
        cf_pool: TypedCounterfactualPool | None = None,
        lambda_dep: float = 0.5,
        alpha: float = 1.0,
        beta: float = 0.2,
        gamma: float = 0.05,
        action_length_norm: bool = True,
        response_token_count: int | None = None,
    ) -> None:
        self.cf_pool = cf_pool or TypedCounterfactualPool()
        self.local_ig_scorer = local_ig_scorer or LocalIGScorer(cf_pool=self.cf_pool)
        self.future_action_ig_scorer = future_action_ig_scorer or FutureActionIGScorer(cf_pool=self.cf_pool)
        self.gate_scorer = gate_scorer or GateRewardScorer()
        self.cost_model = cost_model or ToolCostModel()
        self.lambda_dep = lambda_dep
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.action_length_norm = action_length_norm
        self.response_token_count = response_token_count

    def compute(self, trajectory: Trajectory) -> DAGIGOutput:
        steps = trajectory.steps
        self.cf_pool.add_many(steps)
        g: dict[int, float] = {}
        local_debug: dict[int, dict[str, Any]] = {}
        for step in steps:
            g[step.step_id] = self.local_ig_scorer.score_step(trajectory, step)
            local_debug[step.step_id] = dict(self.local_ig_scorer.last_diagnostics)

        d: dict[tuple[int, int], float] = {}
        future_debug: dict[tuple[int, int], dict[str, Any]] = {}
        for i in range(len(steps) - 1):
            src = steps[i]
            tgt = steps[i + 1]
            edge_value = self.future_action_ig_scorer.score_edge(trajectory, src, tgt)
            d[(src.step_id, tgt.step_id)] = edge_value
            future_debug[(src.step_id, tgt.step_id)] = dict(self.future_action_ig_scorer.last_diagnostics)

        step_ids = [step.step_id for step in steps]
        returns = dagig_lite_returns(g, d, step_ids, self.lambda_dep)
        token_rewards = [0.0 for _ in range(self._token_count(trajectory))]
        step_rewards: list[StepReward] = []

        for idx, step in enumerate(steps):
            future_ig = d.get((step.step_id, steps[idx + 1].step_id), 0.0) if idx < len(steps) - 1 else 0.0
            gate_reward = self.gate_scorer.score(trajectory, step)
            cost_penalty = self.cost_model.score(step)
            propagated_return = returns.get(step.step_id, 0.0)
            total = self.alpha * propagated_return + self.beta * gate_reward - self.gamma * cost_penalty
            inject_span_rewards(token_rewards, step.action_span, total, self.action_length_norm)
            diagnostics = {
                "action_length": max(1, step.action_span[1] - step.action_span[0]),
                "tool_type": step.tool_type,
                "final_correct": trajectory.final_correct,
                "local_debug": local_debug.get(step.step_id, {}),
                "future_debug": future_debug.get((step.step_id, steps[idx + 1].step_id), {}) if idx < len(steps) - 1 else {},
            }
            step_rewards.append(
                StepReward(
                    step_id=step.step_id,
                    tool_type=step.tool_type,
                    local_ig=g.get(step.step_id, 0.0),
                    future_action_ig=future_ig,
                    propagated_return=propagated_return,
                    gate_reward=gate_reward,
                    cost_penalty=cost_penalty,
                    total_step_reward=total,
                    diagnostics=diagnostics,
                )
            )

        diagnostics = {
            "local_ig": g,
            "future_action_ig": {f"{src}->{tgt}": value for (src, tgt), value in d.items()},
            "propagated_return": returns,
            "total_reward": {reward.step_id: reward.total_step_reward for reward in step_rewards},
            "action_length": {step.step_id: max(1, step.action_span[1] - step.action_span[0]) for step in steps},
            "tool_type": {step.step_id: step.tool_type for step in steps},
            "final_correct": trajectory.final_correct,
        }
        return DAGIGOutput(step_rewards=step_rewards, token_rewards=token_rewards, diagnostics=diagnostics)

    def compute_batch(self, trajectories: list[Trajectory]) -> RewardBatch:
        step_rewards: dict[str, list[StepReward]] = {}
        token_rewards: dict[str, list[float]] = {}
        diagnostics: dict[str, Any] = {}
        for trajectory in trajectories:
            output = self.compute(trajectory)
            step_rewards[trajectory.sample_id] = output.step_rewards
            token_rewards[trajectory.sample_id] = output.token_rewards
            diagnostics[trajectory.sample_id] = output.diagnostics
        return RewardBatch(
            trajectories=trajectories,
            step_rewards=step_rewards,
            token_rewards=token_rewards,
            diagnostics=diagnostics,
        )

    def _token_count(self, trajectory: Trajectory) -> int:
        if self.response_token_count is not None:
            return self.response_token_count
        if "response_token_count" in trajectory.metadata:
            return int(trajectory.metadata["response_token_count"])
        max_end = 0
        for step in trajectory.steps:
            max_end = max(max_end, step.action_span[1])
        return max(1, max_end)


def step_rewards_to_dicts(rewards: list[StepReward]) -> list[dict[str, Any]]:
    return [asdict(reward) for reward in rewards]

