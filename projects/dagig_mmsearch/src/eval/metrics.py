from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold_answers: list[str]) -> bool:
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(gold) for gold in gold_answers)


def token_f1(prediction: str, gold_answers: list[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / max(1, len(gold_tokens))
        best = max(best, 2 * precision * recall / max(1e-12, precision + recall))
    return best


def tool_stats(steps: list[dict[str, Any]]) -> dict[str, Any]:
    counts = defaultdict(int)
    for step in steps:
        counts[f"num_{step.get('tool_type')}"] += 1
    total = len(steps)
    counts["num_total_tools"] = total
    return dict(counts)


def aggregate_rollouts(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    n = max(1, len(rows))
    accuracy = sum(bool(row.get("final_correct", False)) for row in rows) / n
    avg_tool_calls = sum(len(row.get("steps", [])) for row in rows) / n
    avg_latency = sum(float(row.get("latency", 0.0)) for row in rows) / n
    invalid = sum(int(row.get("invalid_action", 0)) for row in rows) / n
    success_calls = 0
    total_calls = 0
    for row in rows:
        for step in row.get("steps", []):
            if step.get("tool_type") != "stop":
                total_calls += 1
                success_calls += int(step.get("success", True))
    tool_success = success_calls / max(1, total_calls)
    over_search = sum(len(row.get("steps", [])) > 2 and row.get("final_correct", False) for row in rows) / n
    under_search = sum(len(row.get("steps", [])) <= 1 and not row.get("final_correct", False) for row in rows) / n
    return {
        "method": method,
        "exact_match": accuracy,
        "token_f1": sum(token_f1(row.get("final_answer", ""), row.get("gold_answers", [])) for row in rows) / n,
        "accuracy": accuracy,
        "avg_tool_calls": avg_tool_calls,
        "avg_search_rounds": avg_tool_calls,
        "avg_latency": avg_latency,
        "answer_per_tool_call": accuracy / max(1.0, avg_tool_calls),
        "tool_call_success_rate": tool_success,
        "invalid_action_rate": invalid,
        "over_search_rate": over_search,
        "under_search_rate": under_search,
    }
