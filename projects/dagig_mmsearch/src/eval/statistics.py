from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(var)


def reward_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    local: list[float] = []
    future: list[float] = []
    prop: list[float] = []
    totals: list[float] = []
    by_tool: dict[str, list[float]] = defaultdict(list)
    correct: list[float] = []
    reward_sum: list[float] = []
    for row in rows:
        row_reward = 0.0
        for step in row.get("steps", []):
            local.append(float(step.get("local_ig", 0.0)))
            future.append(float(step.get("future_action_ig", 0.0)))
            prop.append(float(step.get("propagated_return", 0.0)))
            total = float(step.get("total_step_reward", 0.0))
            totals.append(total)
            row_reward += total
            by_tool[step.get("tool_type", "unknown")].append(total)
        correct.append(1.0 if row.get("final_correct", False) else 0.0)
        reward_sum.append(row_reward)
    local_m, local_s = mean_std(local)
    future_m, future_s = mean_std(future)
    prop_m, prop_s = mean_std(prop)
    positive_ratio = sum(value > 0 for value in totals) / max(1, len(totals))
    corr = _correlation(reward_sum, correct)
    result = {
        "local_ig_mean": local_m,
        "local_ig_std": local_s,
        "future_action_ig_mean": future_m,
        "future_action_ig_std": future_s,
        "propagated_return_mean": prop_m,
        "propagated_return_std": prop_s,
        "positive_reward_ratio": positive_ratio,
        "reward_final_correctness_correlation": corr,
    }
    for tool, values in by_tool.items():
        result[f"reward_mean_{tool}"] = mean_std(values)[0]
    return result


def _correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)

