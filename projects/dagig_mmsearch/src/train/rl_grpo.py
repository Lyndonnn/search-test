from __future__ import annotations

import argparse

from agent.rollout import dagig_reward_debug_rollout
from data.schema import toy_samples
from eval.metrics import aggregate_rollouts
from eval.statistics import reward_diagnostics
from train.trainer_utils import apply_oom_downgrade, load_config, save_config
from utils.io import write_csv, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="projects/dagig_mmsearch/configs/dagig_lite_qwen25vl_3b_a800.yaml")
    parser.add_argument("--stage", default="dagig_lite", choices=["outcome", "local_ig", "dagig_lite"])
    parser.add_argument("--simulate-oom-downgrade", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.simulate_oom_downgrade:
        cfg = apply_oom_downgrade(cfg)
        save_config(f"logs/{args.stage}_downgraded_config.yaml", cfg)
    rows = dagig_reward_debug_rollout(toy_samples())
    for row in rows:
        row["method"] = args.stage
        if args.stage == "outcome":
            for step in row.get("steps", []):
                step["local_ig"] = 0.0
                step["future_action_ig"] = 0.0
                step["propagated_return"] = 0.0
        if args.stage == "local_ig":
            for step in row.get("steps", []):
                step["future_action_ig"] = 0.0
                step["propagated_return"] = step.get("local_ig", 0.0)
    output_dir = {
        "outcome": "results/outcome_rl/outcome_rl_smoke.jsonl",
        "local_ig": "results/local_ig/local_ig_smoke.jsonl",
        "dagig_lite": "results/dagig_lite/dagig_lite_smoke.jsonl",
    }[args.stage]
    write_jsonl(output_dir, rows)
    summary = aggregate_rollouts(rows, args.stage)
    summary.update(reward_diagnostics(rows))
    write_json(f"logs/{args.stage}_summary.json", summary)
    write_csv(f"paper_artifacts/tables/{args.stage}_summary.csv", [summary])


if __name__ == "__main__":
    main()

