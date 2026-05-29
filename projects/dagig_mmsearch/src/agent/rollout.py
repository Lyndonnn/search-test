from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from agent.policy_wrapper import SimpleTokenizer
from data.schema import VQASample
from eval.metrics import exact_match, tool_stats
from reward.dag_ig import DAGIGLiteReward
from reward.types import ToolStep, Trajectory
from tools.dispatcher import ToolDispatcher


def _answer_from_observation(raw: Any, fallback: str = "unknown") -> str:
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("answer"):
                return str(item["answer"])
    if isinstance(raw, dict) and raw.get("answer"):
        return str(raw["answer"])
    return fallback


def _make_step(
    step_id: int,
    tool_type: str,
    action_text: str,
    raw_observation: Any,
    evidence_summary: str,
    span_start: int,
    tokenizer: SimpleTokenizer,
    question: str,
    previous_context: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[ToolStep, int, str]:
    action_tokens = tokenizer.encode(action_text) or [abs(hash(tool_type)) % 32000]
    action_span = (span_start, span_start + len(action_tokens))
    context_before = previous_context or f"Question: {question}"
    context_after = context_before + f"\nAction[{step_id}:{tool_type}]: {action_text}\nObservation[{step_id}]: {evidence_summary}"
    step = ToolStep(
        step_id=step_id,
        tool_type=tool_type,
        action_text=action_text,
        action_tokens=action_tokens,
        action_span=action_span,
        raw_observation=raw_observation,
        evidence_summary=evidence_summary,
        context_before_action=context_before,
        context_after_observation=context_after,
        metadata=metadata or {},
    )
    return step, action_span[1], context_after


def direct_vqa_rollout(samples: list[VQASample]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        start = time.time()
        answer = _direct_answer(sample.question)
        correct = exact_match(answer, sample.gold_answers)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "gold_answers": sample.gold_answers,
                "final_answer": answer,
                "final_correct": correct,
                "steps": [],
                "tool_stats": {"num_total_tools": 0},
                "latency": time.time() - start,
                "method": "direct_vqa",
            }
        )
    return rows


def prompted_search_rollout(samples: list[VQASample], search_topk: int = 5) -> tuple[list[dict[str, Any]], list[Trajectory]]:
    dispatcher = ToolDispatcher(search_topk=search_topk)
    tokenizer = SimpleTokenizer()
    rows: list[dict[str, Any]] = []
    trajectories: list[Trajectory] = []
    for sample in samples:
        start = time.time()
        span = 0
        context = f"Question: {sample.question}"
        steps: list[ToolStep] = []
        first_tool = "image_search" if "bridge" in sample.question.lower() else "text_search"
        action = sample.question
        result = dispatcher.run(first_tool, action, topk=search_topk)
        answer = _answer_from_observation(result.raw_observation)
        step0, span, context = _make_step(
            0,
            first_tool,
            action,
            result.raw_observation,
            result.evidence_summary,
            span,
            tokenizer,
            sample.question,
            context,
            {"success": result.success, "answer": answer, "sample_id": sample.sample_id},
        )
        steps.append(step0)
        stop_result = dispatcher.run("stop", answer)
        step1, span, context = _make_step(
            1,
            "stop",
            answer,
            stop_result.raw_observation,
            stop_result.evidence_summary,
            span,
            tokenizer,
            sample.question,
            context,
            {"success": True, "answer": answer, "sample_id": sample.sample_id},
        )
        steps.append(step1)
        correct = exact_match(answer, sample.gold_answers)
        trajectory = Trajectory(
            sample_id=sample.sample_id,
            question=sample.question,
            images=sample.images,
            gold_answers=sample.gold_answers,
            steps=steps,
            final_answer=answer,
            final_correct=correct,
            full_prompt=sample.question,
            full_response="\n".join(step.action_text for step in steps),
            metadata={
                "response_token_count": span,
                "tool_free_answers": [_direct_answer(sample.question)] * 3
                if not sample.metadata.get("needs_search")
                else ["unknown", _direct_answer(sample.question), "uncertain"],
            },
        )
        trajectories.append(trajectory)
        row_steps = [{**asdict(step), "success": step.metadata.get("success", True)} for step in steps]
        rows.append(
            {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "gold_answers": sample.gold_answers,
                "final_answer": answer,
                "final_correct": correct,
                "steps": row_steps,
                "tool_stats": tool_stats(row_steps),
                "latency": time.time() - start,
                "method": "prompted_search",
            }
        )
    return rows, trajectories


def dagig_reward_debug_rollout(samples: list[VQASample]) -> list[dict[str, Any]]:
    rows, trajectories = prompted_search_rollout(samples)
    reward = DAGIGLiteReward(lambda_dep=0.5, alpha=1.0, beta=0.2, gamma=0.05)
    enriched: list[dict[str, Any]] = []
    for row, trajectory in zip(rows, trajectories):
        output = reward.compute(trajectory)
        step_rewards = {step_reward.step_id: step_reward for step_reward in output.step_rewards}
        new_steps = []
        for step in row["steps"]:
            reward_item = step_rewards[step["step_id"]]
            step.update(
                {
                    "local_ig": reward_item.local_ig,
                    "future_action_ig": reward_item.future_action_ig,
                    "propagated_return": reward_item.propagated_return,
                    "gate_reward": reward_item.gate_reward,
                    "cost_penalty": reward_item.cost_penalty,
                    "total_step_reward": reward_item.total_step_reward,
                    "counterfactual_debug": reward_item.diagnostics.get("local_debug", {}).get("counterfactual_debug", []),
                }
            )
            new_steps.append(step)
        row["steps"] = new_steps
        row["token_rewards"] = output.token_rewards
        row["reward_diagnostics"] = output.diagnostics
        row["method"] = "dagig_lite"
        enriched.append(row)
    return enriched


def _direct_answer(question: str) -> str:
    q = question.lower()
    if "eiffel" in q:
        return "Paris"
    if "colosseum" in q:
        return "Rome"
    if "sydney opera" in q:
        return "Sydney"
    return "unknown"
