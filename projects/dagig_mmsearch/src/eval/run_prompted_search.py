from __future__ import annotations

from agent.rollout import prompted_search_rollout
from data.schema import toy_samples
from eval.metrics import aggregate_rollouts
from utils.io import write_csv, write_jsonl


def main() -> None:
    rows, _ = prompted_search_rollout(toy_samples())
    write_jsonl("results/prompted_search/prompted_search_smoke.jsonl", rows)
    write_jsonl("results/baselines/prompted_search_smoke.jsonl", rows)
    write_csv("paper_artifacts/tables/prompted_search.csv", [aggregate_rollouts(rows, "prompted_search")])


if __name__ == "__main__":
    main()

