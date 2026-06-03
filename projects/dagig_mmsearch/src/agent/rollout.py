from __future__ import annotations

import re
import time
from dataclasses import asdict
from typing import Any

from agent.parser import PLACEHOLDER_ACTIONS, ParsedAction, parse_action, parse_final_answer
from agent.policy_wrapper import PolicyWrapper, SimpleTokenizer
from data.schema import VQASample
from eval.metrics import exact_match, tool_stats
from reward.dag_ig import DAGIGLiteReward
from reward.types import ToolStep, Trajectory
from tools.base import summarize_observation
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
        parsed = _repair_placeholder_action(sample, parsed)
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


def model_agent_two_turn_rollout(
    samples: list[VQASample],
    search_topk: int = 5,
    text_index_path: str = "data/indexes/text_corpus.jsonl",
    image_index_path: str = "data/indexes/image_corpus.jsonl",
    policy: PolicyWrapper | None = None,
    max_new_tokens: int = 96,
    answer_max_new_tokens: int = 64,
    temperature: float = 0.0,
    redact_observation_answers: bool = True,
) -> tuple[list[dict[str, Any]], list[Trajectory]]:
    """Two-turn non-oracle rollout: model searches, observes, then answers.

    Unlike `agentic_search_rollout`, this does not read `answer` fields from
    tool outputs. The final answer must be generated by the policy after seeing
    an answer-hidden evidence summary.
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
        direct_answer = _direct_answer(sample.question)
        needs_search = bool(sample.metadata.get("needs_search", direct_answer == "unknown"))
        first = _repair_placeholder_action(
            sample,
            _first_agent_action(
                sample,
                policy,
                needs_search,
                direct_answer,
                scripted_direct_stop=False,
                force_search_when_needed=False,
                fallback_on_invalid=False,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            ),
        )
        first_invalid = int((not first.valid) or first.tool_type not in {"text_search", "image_search", "stop"})
        final_invalid = 0

        if first.tool_type in {"text_search", "image_search"} and not first_invalid:
            result = dispatcher.run(first.tool_type, first.action_text or sample.question, topk=search_topk)
            safe_raw = _strip_answer_fields(result.raw_observation)
            if redact_observation_answers:
                safe_raw = _redact_answer_text(safe_raw, sample.gold_answers)
            safe_summary = summarize_observation(first.tool_type, safe_raw, dispatcher.max_summary_tokens)
            step0, span, context = _make_step(
                0,
                first.tool_type,
                first.action_text or sample.question,
                safe_raw,
                safe_summary,
                span,
                tokenizer,
                sample.question,
                context,
                {
                    "success": result.success,
                    "sample_id": sample.sample_id,
                    "raw_policy_output": first.arguments.get("raw_policy_output", ""),
                    "non_oracle": True,
                },
            )
            steps.append(step0)
            final = _final_answer_action(sample, policy, context, answer_max_new_tokens, temperature)
            final = _repair_placeholder_action(sample, final)
            final_invalid = int((not final.valid) or final.tool_type != "stop")
            answer = final.action_text if final.tool_type == "stop" and final.action_text else final.arguments.get("raw_policy_output", "")
            answer = answer or "unknown"
            step1, span, context = _make_stop_step(
                sample,
                answer,
                span,
                tokenizer,
                context,
                step_id=1,
                metadata={
                    "success": True,
                    "sample_id": sample.sample_id,
                    "raw_policy_output": final.arguments.get("raw_policy_output", ""),
                    "non_oracle": True,
                },
            )
            steps.append(step1)
        else:
            if first.tool_type == "stop" and first.action_text:
                answer = first.action_text
            else:
                answer = first.arguments.get("raw_policy_output", "") or first.action_text or "unknown"
            step0, span, context = _make_stop_step(
                sample,
                answer,
                span,
                tokenizer,
                context,
                step_id=0,
                metadata={
                    "success": True,
                    "sample_id": sample.sample_id,
                    "raw_policy_output": first.arguments.get("raw_policy_output", ""),
                    "non_oracle": True,
                },
            )
            steps.append(step0)

        final_answer = steps[-1].action_text if steps else ""
        correct = exact_match(final_answer, sample.gold_answers)
        trajectory = Trajectory(
            sample_id=sample.sample_id,
            question=sample.question,
            images=sample.images,
            gold_answers=sample.gold_answers,
            steps=steps,
            final_answer=final_answer,
            final_correct=correct,
            full_prompt=sample.question,
            full_response="\n".join(step.action_text for step in steps),
            metadata={
                "response_token_count": span,
                "tool_free_answers": _tool_free_probe_answers(sample, direct_answer, needs_search),
                "rollout_mode": "model_agent_two_turn",
                "non_oracle": True,
            },
        )
        trajectories.append(trajectory)
        row_steps = [{**asdict(step), "success": step.metadata.get("success", True)} for step in steps]
        rows.append(
            {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "gold_answers": sample.gold_answers,
                "final_answer": final_answer,
                "final_correct": correct,
                "steps": row_steps,
                "tool_stats": tool_stats(row_steps),
                "latency": time.time() - start,
                "method": "model_agent_two_turn",
                "invalid_action": int(first_invalid or final_invalid),
                "invalid_first_action": first_invalid,
                "invalid_final_action": final_invalid,
                "policy_action": first.arguments,
                "non_oracle": True,
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
        "Return exactly one compact JSON object and no prose.\n"
        "Allowed tools: text_search, image_search, stop.\n"
        "For search, action must be a concrete query copied from the question, not the word query.\n"
        "For stop, action must be the final answer, not the words final answer.\n"
        "If you are uncertain, choose text_search.\n"
        f"Question: {sample.question}\n"
        "JSON only:"
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


def _repair_placeholder_action(sample: VQASample, parsed: ParsedAction) -> ParsedAction:
    action = parsed.action_text.strip()
    if parsed.tool_type in {"text_search", "image_search"} and action.lower() in PLACEHOLDER_ACTIONS:
        args = dict(parsed.arguments)
        args["original_action"] = parsed.action_text
        args["action_repair"] = "placeholder_to_question"
        return ParsedAction(parsed.tool_type, sample.question, args, parsed.valid, parsed.error)
    if parsed.tool_type == "stop" and action.lower() in PLACEHOLDER_ACTIONS:
        args = dict(parsed.arguments)
        args["original_action"] = parsed.action_text
        args["action_repair"] = "placeholder_to_unknown"
        return ParsedAction("stop", "unknown", args, False, parsed.error or "placeholder_stop_action")
    return parsed


def _final_answer_action(
    sample: VQASample,
    policy: PolicyWrapper,
    context_after_observation: str,
    max_new_tokens: int,
    temperature: float,
) -> ParsedAction:
    prompt = (
        "You are a multimodal search agent.\n"
        "Use the observation to answer the question.\n"
        "Return exactly one compact JSON object and no prose.\n"
        'Allowed form: {"action":"stop","answer":"final answer"}.\n'
        "If the observation is insufficient, answer unknown.\n\n"
        f"{context_after_observation}\n"
        "JSON only:"
    )
    output = policy.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    parsed = parse_final_answer(output.text)
    parsed.arguments.setdefault("raw_policy_output", output.text)
    parsed.arguments.setdefault("policy_metadata", output.metadata)
    return parsed


def _strip_answer_fields(raw: Any) -> Any:
    blocked = {"answer", "answers", "gold_answer", "gold_answers", "ground_truth", "target", "label"}
    if isinstance(raw, dict):
        return {key: _strip_answer_fields(value) for key, value in raw.items() if str(key).lower() not in blocked}
    if isinstance(raw, list):
        return [_strip_answer_fields(item) for item in raw]
    if isinstance(raw, tuple):
        return tuple(_strip_answer_fields(item) for item in raw)
    return raw


def _redact_answer_text(raw: Any, gold_answers: list[str]) -> Any:
    if isinstance(raw, dict):
        return {key: _redact_answer_text(value, gold_answers) for key, value in raw.items()}
    if isinstance(raw, list):
        return [_redact_answer_text(item, gold_answers) for item in raw]
    if isinstance(raw, tuple):
        return tuple(_redact_answer_text(item, gold_answers) for item in raw)
    if isinstance(raw, str):
        text = raw
        text = re.sub(r"(?i)\banswer\s*:\s*[^.。\n]+[.。]?", "Answer: [hidden]. ", text)
        for answer in gold_answers:
            answer = str(answer).strip()
            if answer:
                text = re.sub(re.escape(answer), "[hidden]", text, flags=re.IGNORECASE)
        return text
    return raw


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
    metadata: dict[str, Any] | None = None,
) -> tuple[ToolStep, int, str]:
    dispatcher = ToolDispatcher()
    stop_result = dispatcher.run("stop", answer)
    step_metadata = {"success": True, "answer": answer, "sample_id": sample.sample_id}
    if metadata:
        step_metadata.update(metadata)
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
        step_metadata,
    )
