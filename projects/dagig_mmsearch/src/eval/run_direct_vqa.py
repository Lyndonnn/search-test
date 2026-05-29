from __future__ import annotations

from agent.rollout import direct_vqa_rollout
from data.schema import toy_samples
from eval.metrics import aggregate_rollouts
from utils.io import write_csv, write_jsonl


def main() -> None:
    rows = direct_vqa_rollout(toy_samples())
    write_jsonl("results/direct_vqa/direct_vqa_smoke.jsonl", rows)
    write_jsonl("results/baselines/direct_vqa_smoke.jsonl", rows)
    write_csv("paper_artifacts/tables/direct_vqa.csv", [aggregate_rollouts(rows, "direct_vqa")])


if __name__ == "__main__":
    main()

