from __future__ import annotations

from reward.typed_pool import TypedCounterfactualPool
from reward.types import CounterfactualObservation, ToolStep, Trajectory


def build_context(
    trajectory: Trajectory,
    replace_step_id: int | None = None,
    replacement: CounterfactualObservation | None = None,
    before_step_id: int | None = None,
) -> str:
    lines = [f"Question: {trajectory.question}"]
    if trajectory.images:
        lines.append("Images: " + ", ".join(trajectory.images))
    for step in trajectory.steps:
        if before_step_id is not None and step.step_id >= before_step_id:
            break
        lines.append(f"Action[{step.step_id}:{step.tool_type}]: {step.action_text}")
        if replace_step_id == step.step_id and replacement is not None:
            summary = replacement.evidence_summary
        else:
            summary = step.evidence_summary
        lines.append(f"Observation[{step.step_id}]: {summary}")
    return "\n".join(lines)


def sample_counterfactuals(
    cf_pool: TypedCounterfactualPool, step: ToolStep, k: int
) -> list[CounterfactualObservation]:
    return cf_pool.sample(step.tool_type, step, k)

