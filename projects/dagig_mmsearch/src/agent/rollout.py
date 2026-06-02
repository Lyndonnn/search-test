from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from agent.parser import ParsedAction, parse_action
from agent.policy_wrapper import PolicyWrapper, SimpleTokenizer
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


def prompted_search_rollout(
    samples: list[VQASample],
    search_topk: int = 5,
    text_index_path: str = "data/indexes/text_corpus.jsonl",
    image_index_path: str = "data/indexes/image_corpus.jsonl",
) -> tuple[list[dict[str, Any]], list[Trajectory]]:
    dispatcher = ToolDispatcher(
        search_topk=search_topk,
        text_index_path=text_index_path,
        image_index_path=image_index_path,
    )
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


def agentic_search_rollout(
    samples: list[VQASample],
    search_topk: int = 5,
    text_index_path: str = "data/indexes/text_corpus.jsonl",
    image_index_path: str = "data/indexes/image_corpus.jsonl",
    policy: PolicyWrapper | None = None,
    scripted_direct_stop: bool = True,
    force_search_when_needed: bool = True,
    fallback_on_invalid: bool = True,
    max_new_tokens: int = 96,
    temperature: float = 0.0,
) -> tuple[list[dict[str, Any]], list[Trajectory]]:
    """A lightweight rollout path where the first action can be stop or search.

    This is intentionally not a training rollout. It is a controlled smoke test
    for over-search and under-search diagnostics before connecting veRL/MMSearch.
    """

    dispatcher = ToolDispatcher(
        search_topk=search_topk,
        text_index_path=text_index_path,
        image_index_path=image_index_path,
    )
    tokenizer = SimpleTokenizer()
    policy = policy or PolicyWrapper(tokenizer=tokenizer)
    rows: list[dict[str, Any]] = []
    trajectories: list[Trajectory] = []
    for sample in samples:
        start = time.time()
        span = 0
        context = f"Question: {sample.question}"
        steps: list[ToolStep] = []
        invalid_action = 0
        direct_answer = _direct_answer(sample.question)
        needs_search = bool(sample.metadata.get("needs_search", direct_answer == "unknown"))

        parsed = _first_agent_action(
            sample,
            policy,
            needs_search,
            direct_answer,
            scripted_direct_stop=scripted_direct_stop,
            force_search_when_needed=force_search_when_needed,
            fallback_on_invalid=fallback_on_invalid,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        allowed_tools = {"text_search", "image_search", "stop"}
        invalid_action = int((not parsed.valid) or parsed.tool_type not in allowed_tools)
        if parsed.tool_type == "stop":
            answer = parsed.action_text or direct_answer
            step, span, context = _make_stop_step(sample, answer, span, tokenizer, context)
            steps.append(step)
        elif parsed.tool_type in {"text_search", "image_search"}:
            tool_type = parsed.tool_type if parsed.tool_type in {"text_search", "image_search"} else _default_search_tool(sample)
            action = parsed.action_text or _default_search_query(sample)
            result = dispatcher.run(tool_type, action, topk=search_topk)
            answer = _answer_from_observation(result.raw_observation, fallback=direct_answer if direct_answer != "unknown" else "unknown")
            step0, span, context = _make_step(
                0,
                tool_type,
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
            step1, span, context = _make_stop_step(sample, answer, span, tokenizer, context, step_id=1)
            steps.append(step1)
        else:
            answer = parsed.action_text or "unknown"
            step, span, context = _make_stop_step(sample, answer, span, tokenizer, context)
            steps.append(step)

        correct = exact_match(steps[-1].action_text if steps else "", sample.gold_answers)
        trajectory = Trajectory(
            sample_id=sample.sample_id,
            question=sample.question,
            images=sample.images,
            gold_answers=sample.gold_answers,
            steps=steps,
            final_answer=steps[-1].action_text if steps else "",
            final_correct=correct,
            full_prompt=sample.question,
            full_response="\n".join(step.action_text for step in steps),
            metadata={
                "response_token_count": span,
                "tool_free_answers": _tool_free_probe_answers(sample, direct_answer, needs_search),
                "rollout_mode": "agentic_search",
            },
        )
        trajectories.append(trajectory)
        row_steps = [{**asdict(step), "success": step.metadata.get("success", True)} for step in steps]
        rows.append(
            {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "gold_answers": sample.gold_answers,
                "final_answer": trajectory.final_answer,
                "final_correct": correct,
                "steps": row_steps,
                "tool_stats": tool_stats(row_steps),
                "latency": time.time() - start,
                "method": "agentic_search",
                "invalid_action": invalid_action,
                "policy_action": parsed.arguments,
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


def _first_agent_action(
    sample: VQASample,
    policy: PolicyWrapper,
    needs_search: bool,
    direct_answer: str,
    scripted_direct_stop: bool = True,
    force_search_when_needed: bool = True,
    fallback_on_invalid: bool = True,
    max_new_tokens: int = 96,
    temperature: float = 0.0,
) -> ParsedAction:
    if scripted_direct_stop and not needs_search and direct_answer != "unknown":
        return ParsedAction("stop", direct_answer, {"answer": direct_answer, "source": "tool_free"}, True)
    prompt = (
        "You are a multimodal search agent.\n"
        "Return exactly one JSON object and no extra text.\n"
        'Allowed forms: {"tool":"text_search","action":"query"}, '
        '{"tool":"image_search","action":"visual query"}, '
        '{"tool":"stop","action":"final answer"}.\n'
        "Use search if the answer is not directly known.\n"
        f"Question: {sample.question}"
    )
    output = policy.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    parsed = parse_action(output.text)
    parsed.arguments.setdefault("raw_policy_output", output.text)
    parsed.arguments.setdefault("policy_metadata", output.metadata)
    if (
        force_search_when_needed
        and parsed.tool_type == "stop"
        and (not parsed.action_text or parsed.action_text == "unknown")
        and needs_search
    ):
        return ParsedAction(_default_search_tool(sample), _default_search_query(sample), {"source": "needs_search_fallback"}, True)
    if fallback_on_invalid and parsed.tool_type not in {"text_search", "image_search", "stop"}:
        return ParsedAction(_default_search_tool(sample), _default_search_query(sample), {"source": "invalid_tool_fallback"}, False)
    return parsed


def _default_search_tool(sample: VQASample) -> str:
    question = sample.question.lower()
    image_terms = ("image", "shown", "pictured", "photo", "visual", "bridge")
    return "image_search" if any(term in question for term in image_terms) or sample.images else "text_search"


def _default_search_query(sample: VQASample) -> str:
    return sample.question


def _tool_free_probe_answers(sample: VQASample, direct_answer: str, needs_search: bool) -> list[str]:
    if not needs_search and direct_answer != "unknown":
        return [direct_answer, direct_answer, direct_answer]
    return ["unknown", direct_answer, "uncertain"]


def _make_stop_step(
    sample: VQASample,
    answer: str,
    span: int,
    tokenizer: SimpleTokenizer,
    context: str,
    step_id: int = 0,
) -> tuple[ToolStep, int, str]:
    dispatcher = ToolDispatcher()
    stop_result = dispatcher.run("stop", answer)
    return _make_step(
        step_id,
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
