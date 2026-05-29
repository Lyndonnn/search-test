from __future__ import annotations


def dagig_lite_returns(
    local_ig: dict[int, float],
    future_edges: dict[tuple[int, int], float],
    step_ids: list[int],
    lambda_dep: float,
) -> dict[int, float]:
    if not step_ids:
        return {}
    returns: dict[int, float] = {step_ids[-1]: local_ig.get(step_ids[-1], 0.0)}
    for idx in reversed(range(len(step_ids) - 1)):
        src = step_ids[idx]
        tgt = step_ids[idx + 1]
        dep = future_edges.get((src, tgt), 0.0)
        future_utility = max(local_ig.get(tgt, 0.0), 0.0)
        returns[src] = local_ig.get(src, 0.0) + lambda_dep * dep * future_utility
    return returns


def inject_span_rewards(
    token_rewards: list[float],
    action_span: tuple[int, int],
    total_reward: float,
    length_norm: bool = True,
) -> None:
    start, end = action_span
    start = max(0, start)
    end = min(len(token_rewards), max(start, end))
    if end <= start:
        return
    denom = max(1, end - start) if length_norm else 1
    bonus = total_reward / denom
    for position in range(start, end):
        token_rewards[position] += bonus

