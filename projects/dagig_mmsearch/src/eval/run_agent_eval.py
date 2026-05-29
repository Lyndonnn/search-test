from __future__ import annotations

from agent.rollout import dagig_reward_debug_rollout
from data.schema import toy_samples
from eval.metrics import aggregate_rollouts
from eval.statistics import reward_diagnostics
from utils.io import write_csv, write_jsonl


def main() -> None:
    rows = dagig_reward_debug_rollout(toy_samples())
    write_jsonl("results/dagig_lite/dagig_reward_debug.jsonl", rows)
    main_row = aggregate_rollouts(rows, "dagig_lite")
    main_row.update(reward_diagnostics(rows))
    write_csv("paper_artifacts/tables/dagig_lite_debug.csv", [main_row])


if __name__ == "__main__":
    main()

