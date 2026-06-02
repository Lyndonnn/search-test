from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.rollout import agentic_search_rollout, prompted_search_rollout
from data.dataset_mixer import read_samples_jsonl
from data.schema import toy_samples
from eval.metrics import aggregate_rollouts
from eval.statistics import reward_diagnostics
from reward.dag_ig import DAGIGLiteReward
from reward.future_action_ig import FutureActionIGScorer
from reward.local_ig import LocalIGScorer
from reward.propagation import dagig_lite_returns, inject_span_rewards
from reward.typed_pool import TypedCounterfactualPool
from reward.types import DAGIGOutput, StepReward, Trajectory
from train.trainer_utils import load_config
from utils.gpu_check import main as print_gpu_check
from utils.hf_reference import load_reference_policy_from_config
from utils.io import ensure_dir, write_csv, write_jsonl


@dataclass(frozen=True)
class AblationVariant:
    name: str
    lambda_dep: float
    alpha: float
    beta: float
    gamma: float
    use_future: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DAG-IG reference-policy reward ablations on a controlled rollout set.")
    parser.add_argument("--config", default="projects/dagig_mmsearch/configs/dagig_lite_qwen25vl_3b_a800.yaml")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--cf-samples", type=int, default=4)
    parser.add_argument("--samples-jsonl", default="")
    parser.add_argument("--text-index", default="data/indexes/text_corpus.jsonl")
    parser.add_argument("--image-index", default="data/indexes/image_corpus.jsonl")
    parser.add_argument("--rollout-mode", choices=["prompted", "agentic"], default="prompted")
    parser.add_argument("--output-dir", default="results/ablations")
    parser.add_argument("--table-output", default="paper_artifacts/tables/reference_ablation.csv")
    parser.add_argument("--delta-output", default="paper_artifacts/tables/reference_ablation_delta.csv")
    parser.add_argument("--method-prefix", default="reference_ablation")
    parser.add_argument(
        "--variants",
        default="local_ig_only,dagig_lite,dagig_no_gate,dagig_no_cost,lambda_0,lambda_025,lambda_05,lambda_1",
    )
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
    if args.rollout_mode == "agentic":
        base_rows, trajectories = agentic_search_rollout(
            samples[: args.limit],
            text_index_path=args.text_index,
            image_index_path=args.image_index,
        )
    else:
        base_rows, trajectories = prompted_search_rollout(
            samples[: args.limit],
            text_index_path=args.text_index,
            image_index_path=args.image_index,
        )

    base_outputs = [reward.compute(trajectory) for trajectory in trajectories]
    variants = build_variants(args.variants, cfg)
    output_dir = ensure_dir(args.output_dir)
    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        method = f"{args.method_prefix}_{variant.name}"
        rows = materialize_variant_rows(base_rows, trajectories, base_outputs, variant, method)
        rows_by_variant[variant.name] = rows
        write_jsonl(Path(output_dir) / f"{variant.name}.jsonl", rows)
        summary = aggregate_rollouts(rows, method)
        summary.update(reward_diagnostics(rows))
        summary.update(
            {
                "rollout_mode": args.rollout_mode,
                "cf_samples": args.cf_samples,
                "lambda_dep": variant.lambda_dep,
                "alpha": variant.alpha,
                "beta": variant.beta,
                "gamma": variant.gamma,
                "use_future": variant.use_future,
            }
        )
        summaries.append(summary)
        all_rows.extend(rows)

    write_jsonl(Path(output_dir) / "reference_ablation_all.jsonl", all_rows)
    write_csv(args.table_output, summaries)
    if "local_ig_only" in rows_by_variant and "dagig_lite" in rows_by_variant:
        delta_rows = build_delta_rows(rows_by_variant["local_ig_only"], rows_by_variant["dagig_lite"])
        write_csv(args.delta_output, delta_rows)
    print(f"saved ablation rows to {output_dir}")
    print(f"saved ablation table to {args.table_output}")
    print(f"saved delta table to {args.delta_output}")


def build_variants(raw_names: str, cfg: dict[str, Any]) -> list[AblationVariant]:
    reward_cfg = cfg.get("reward", {})
    alpha = float(reward_cfg.get("local_ig_weight", 0.4))
    beta = float(reward_cfg.get("gate_weight", 0.2))
    gamma = float(reward_cfg.get("cost_weight", 0.05))
    base_lambda = float(reward_cfg.get("lambda_dep", 0.5))
    variants = []
    for raw_name in [item.strip() for item in raw_names.split(",") if item.strip()]:
        name = raw_name.lower()
        if name in {"local", "local_ig", "local_ig_only", "no_future"}:
            variants.append(AblationVariant("local_ig_only", 0.0, alpha, beta, gamma, use_future=False))
        elif name in {"dagig", "dagig_lite", "full"}:
            variants.append(AblationVariant("dagig_lite", base_lambda, alpha, beta, gamma, use_future=True))
        elif name in {"dagig_no_gate", "no_gate"}:
            variants.append(AblationVariant("dagig_no_gate", base_lambda, alpha, 0.0, gamma, use_future=True))
        elif name in {"dagig_no_cost", "no_cost"}:
            variants.append(AblationVariant("dagig_no_cost", base_lambda, alpha, beta, 0.0, use_future=True))
        elif name.startswith("lambda_"):
            value = _parse_lambda(name)
            variants.append(AblationVariant(name, value, alpha, beta, gamma, use_future=True))
        else:
            raise ValueError(f"Unknown ablation variant: {raw_name}")
    deduped = []
    seen = set()
    for variant in variants:
        if variant.name not in seen:
            deduped.append(variant)
            seen.add(variant.name)
    return deduped


def materialize_variant_rows(
    base_rows: list[dict[str, Any]],
    trajectories: list[Trajectory],
    base_outputs: list[DAGIGOutput],
    variant: AblationVariant,
    method: str,
) -> list[dict[str, Any]]:
    rows = copy.deepcopy(base_rows)
    for row, trajectory, output in zip(rows, trajectories, base_outputs):
        base_rewards = {item.step_id: item for item in output.step_rewards}
        step_ids = [step.step_id for step in trajectory.steps]
        g = {item.step_id: item.local_ig for item in output.step_rewards}
        d = _edge_values(output.step_rewards, step_ids, use_future=variant.use_future)
        returns = dagig_lite_returns(g, d, step_ids, variant.lambda_dep)
        token_rewards = [0.0 for _ in range(_token_count(trajectory))]
        updated_rewards: list[StepReward] = []
        for idx, step in enumerate(trajectory.steps):
            base_reward = base_rewards[step.step_id]
            edge = (step.step_id, trajectory.steps[idx + 1].step_id) if idx < len(trajectory.steps) - 1 else None
            future_ig = d.get(edge, 0.0) if edge else 0.0
            propagated_return = returns.get(step.step_id, 0.0)
            total = variant.alpha * propagated_return + variant.beta * base_reward.gate_reward - variant.gamma * base_reward.cost_penalty
            inject_span_rewards(token_rewards, step.action_span, total, length_norm=True)
            updated_rewards.append(
                StepReward(
                    step_id=step.step_id,
                    tool_type=step.tool_type,
                    local_ig=base_reward.local_ig,
                    future_action_ig=future_ig,
                    propagated_return=propagated_return,
                    gate_reward=base_reward.gate_reward,
                    cost_penalty=base_reward.cost_penalty,
                    total_step_reward=total,
                    diagnostics={
                        **base_reward.diagnostics,
                        "ablation_variant": variant.name,
                        "use_future": variant.use_future,
                        "lambda_dep": variant.lambda_dep,
                    },
                )
            )
        by_step = {item.step_id: item for item in updated_rewards}
        for step in row.get("steps", []):
            reward_item = by_step[step["step_id"]]
            step.update(
                {
                    "local_ig": reward_item.local_ig,
                    "future_action_ig": reward_item.future_action_ig,
                    "propagated_return": reward_item.propagated_return,
                    "gate_reward": reward_item.gate_reward,
                    "cost_penalty": reward_item.cost_penalty,
                    "total_step_reward": reward_item.total_step_reward,
                    "reward_diagnostics": asdict(reward_item),
                }
            )
        row["method"] = method
        row["token_rewards"] = token_rewards
        row["reward_diagnostics"] = {
            "local_ig": g,
            "future_action_ig": {f"{src}->{tgt}": value for (src, tgt), value in d.items()},
            "propagated_return": returns,
            "total_reward": {item.step_id: item.total_step_reward for item in updated_rewards},
            "action_length": {step.step_id: max(1, step.action_span[1] - step.action_span[0]) for step in trajectory.steps},
            "tool_type": {step.step_id: step.tool_type for step in trajectory.steps},
            "final_correct": trajectory.final_correct,
            "ablation_variant": variant.name,
        }
    return rows


def build_delta_rows(local_rows: list[dict[str, Any]], dagig_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local_row, dagig_row in zip(local_rows, dagig_rows):
        local_steps = {int(step["step_id"]): step for step in local_row.get("steps", [])}
        dagig_steps = {int(step["step_id"]): step for step in dagig_row.get("steps", [])}
        ordered_ids = sorted(dagig_steps)
        for idx, step_id in enumerate(ordered_ids):
            dagig_step = dagig_steps[step_id]
            local_step = local_steps.get(step_id, {})
            next_step_id = ordered_ids[idx + 1] if idx + 1 < len(ordered_ids) else None
            next_step = dagig_steps.get(next_step_id, {}) if next_step_id is not None else {}
            total_delta = float(dagig_step.get("total_step_reward", 0.0)) - float(local_step.get("total_step_reward", 0.0))
            propagated_delta = float(dagig_step.get("propagated_return", 0.0)) - float(local_step.get("propagated_return", 0.0))
            future_ig = float(dagig_step.get("future_action_ig", 0.0))
            next_local_ig = float(next_step.get("local_ig", 0.0)) if next_step else 0.0
            rows.append(
                {
                    "sample_id": dagig_row.get("sample_id", ""),
                    "question": dagig_row.get("question", ""),
                    "final_correct": dagig_row.get("final_correct", False),
                    "final_answer": dagig_row.get("final_answer", ""),
                    "step_id": step_id,
                    "tool_type": dagig_step.get("tool_type", ""),
                    "target_step_id": next_step_id if next_step_id is not None else "",
                    "target_tool_type": next_step.get("tool_type", "") if next_step else "",
                    "local_ig": float(dagig_step.get("local_ig", 0.0)),
                    "future_action_ig": future_ig,
                    "next_local_ig": next_local_ig,
                    "future_edge_active": future_ig > 0.0,
                    "future_credit_eligible": future_ig > 0.0 and next_local_ig > 0.0,
                    "local_only_propagated_return": float(local_step.get("propagated_return", 0.0)),
                    "dagig_propagated_return": float(dagig_step.get("propagated_return", 0.0)),
                    "propagated_delta": propagated_delta,
                    "local_only_total_reward": float(local_step.get("total_step_reward", 0.0)),
                    "dagig_total_reward": float(dagig_step.get("total_step_reward", 0.0)),
                    "total_reward_delta": total_delta,
                    "gate_reward": float(dagig_step.get("gate_reward", 0.0)),
                    "cost_penalty": float(dagig_step.get("cost_penalty", 0.0)),
                    "action_length": _action_length(dagig_step),
                    "is_tool_step": dagig_step.get("tool_type", "") != "stop",
                }
            )
    return rows


def _edge_values(step_rewards: list[StepReward], step_ids: list[int], use_future: bool) -> dict[tuple[int, int], float]:
    if not use_future:
        return {(src, tgt): 0.0 for src, tgt in zip(step_ids[:-1], step_ids[1:])}
    by_step = {item.step_id: item for item in step_rewards}
    return {(src, tgt): by_step[src].future_action_ig for src, tgt in zip(step_ids[:-1], step_ids[1:])}


def _token_count(trajectory: Trajectory) -> int:
    if "response_token_count" in trajectory.metadata:
        return int(trajectory.metadata["response_token_count"])
    return max(1, max((step.action_span[1] for step in trajectory.steps), default=1))


def _parse_lambda(name: str) -> float:
    suffix = name.removeprefix("lambda_")
    if suffix in {"0", "00"}:
        return 0.0
    if suffix in {"025", "0_25"}:
        return 0.25
    if suffix in {"05", "050", "0_5"}:
        return 0.5
    if suffix in {"1", "10", "100"}:
        return 1.0
    return float(suffix.replace("_", "."))


def _action_length(step: dict[str, Any]) -> int:
    span = step.get("action_span", [0, 0])
    if isinstance(span, (list, tuple)) and len(span) == 2:
        return max(1, int(span[1]) - int(span[0]))
    return max(1, len(str(step.get("action_text", "")).split()))


if __name__ == "__main__":
    main()
