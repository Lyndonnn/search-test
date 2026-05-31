from __future__ import annotations

import argparse
from dataclasses import asdict

from agent.rollout import prompted_search_rollout
from data.dataset_mixer import read_samples_jsonl
from data.schema import toy_samples
from eval.metrics import aggregate_rollouts
from eval.statistics import reward_diagnostics
from reward.dag_ig import DAGIGLiteReward
from reward.future_action_ig import FutureActionIGScorer
from reward.local_ig import LocalIGScorer
from reward.typed_pool import TypedCounterfactualPool
from train.trainer_utils import load_config
from utils.gpu_check import main as print_gpu_check
from utils.hf_reference import load_reference_policy_from_config
from utils.io import write_csv, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a tiny DAG-IG batch with a real frozen HF reference policy.")
    parser.add_argument("--config", default="projects/dagig_mmsearch/configs/dagig_lite_qwen25vl_3b_a800.yaml")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--cf-samples", type=int, default=1)
    parser.add_argument("--output", default="results/dagig_lite/reference_logprob_smoke.jsonl")
    parser.add_argument("--samples-jsonl", default="")
    parser.add_argument("--text-index", default="data/indexes/text_corpus.jsonl")
    parser.add_argument("--image-index", default="data/indexes/image_corpus.jsonl")
    parser.add_argument("--method", default="reference_logprob_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    print_gpu_check()
    policy = load_reference_policy_from_config(cfg)
    cf_pool = TypedCounterfactualPool()
    reward = DAGIGLiteReward(
        cf_pool=cf_pool,
        local_ig_scorer=LocalIGScorer(
            model=policy,
            cf_pool=cf_pool,
            cf_samples=args.cf_samples,
            dead_zone=float(cfg.get("reward", {}).get("dead_zone", 0.02)),
            negative_scale=float(cfg.get("reward", {}).get("negative_scale", 0.25)),
        ),
        future_action_ig_scorer=FutureActionIGScorer(
            model=policy,
            cf_pool=cf_pool,
            cf_samples=args.cf_samples,
            dead_zone=float(cfg.get("reward", {}).get("dead_zone", 0.02)),
            positive_only=bool(cfg.get("reward", {}).get("positive_only_future", True)),
        ),
        lambda_dep=float(cfg.get("reward", {}).get("lambda_dep", 0.5)),
        alpha=float(cfg.get("reward", {}).get("local_ig_weight", 0.4)),
        beta=float(cfg.get("reward", {}).get("gate_weight", 0.2)),
        gamma=float(cfg.get("reward", {}).get("cost_weight", 0.05)),
    )

    samples = read_samples_jsonl(args.samples_jsonl) if args.samples_jsonl else toy_samples()
    rows, trajectories = prompted_search_rollout(
        samples[: args.limit],
        text_index_path=args.text_index,
        image_index_path=args.image_index,
    )
    enriched = []
    for row, trajectory in zip(rows, trajectories):
        output = reward.compute(trajectory)
        by_step = {item.step_id: item for item in output.step_rewards}
        for step in row["steps"]:
            step_reward = by_step[step["step_id"]]
            step.update(
                {
                    "local_ig": step_reward.local_ig,
                    "future_action_ig": step_reward.future_action_ig,
                    "propagated_return": step_reward.propagated_return,
                    "gate_reward": step_reward.gate_reward,
                    "cost_penalty": step_reward.cost_penalty,
                    "total_step_reward": step_reward.total_step_reward,
                    "reward_diagnostics": asdict(step_reward),
                }
            )
        row["method"] = args.method
        row["token_rewards"] = output.token_rewards
        row["reward_diagnostics"] = output.diagnostics
        enriched.append(row)

    write_jsonl(args.output, enriched)
    summary = aggregate_rollouts(enriched, args.method)
    summary.update(reward_diagnostics(enriched))
    write_csv("paper_artifacts/tables/reference_logprob_smoke.csv", [summary])
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
