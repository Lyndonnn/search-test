from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolStep:
    step_id: int
    tool_type: str
    action_text: str
    action_tokens: list[int]
    action_span: tuple[int, int]
    raw_observation: Any
    evidence_summary: str
    context_before_action: str
    context_after_observation: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    sample_id: str
    question: str
    images: list[str]
    gold_answers: list[str]
    steps: list[ToolStep]
    final_answer: str
    final_correct: bool
    full_prompt: str
    full_response: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepReward:
    step_id: int
    tool_type: str
    local_ig: float
    future_action_ig: float
    propagated_return: float
    gate_reward: float
    cost_penalty: float
    total_step_reward: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardBatch:
    trajectories: list[Trajectory]
    step_rewards: dict[str, list[StepReward]]
    token_rewards: dict[str, list[float]]
    diagnostics: dict[str, Any]


@dataclass
class CounterfactualObservation:
    raw_observation: Any
    evidence_summary: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DAGIGOutput:
    step_rewards: list[StepReward]
    token_rewards: list[float]
    diagnostics: dict[str, Any]

