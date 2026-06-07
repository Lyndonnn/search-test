from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent.policy_wrapper import SimpleTokenizer
from agent.rollout import _make_step
from data.dataset_mixer import read_samples_jsonl
from data.schema import VQASample, toy_samples
from eval.metrics import exact_match
from reward.dag_ig import DAGIGLiteReward
from reward.future_action_ig import FutureActionIGScorer
from reward.local_ig import LocalIGScorer
from reward.typed_pool import TypedCounterfactualPool
from reward.types import Trajectory
from tools.base import summarize_observation
from tools.dispatcher import ToolDispatcher
from train.trainer_utils import load_config
from utils.gpu_check import main as print_gpu_check
from utils.hf_reference import load_reference_policy_from_config
from utils.io import write_csv, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build teacher multimodal dependency trajectories and offline relabel them with DAG-IG. "
            "This is a diagnostic gate before online RL."
        )
    )
    parser.add_argument("--config", default="projects/dagig_mmsearch/configs/dagig_lite_qwen25vl_3b_a800.yaml")
    parser.add_argument("--samples-jsonl", default="")
    parser.add_argument("--parquet", default="mmsearch_r1/data/fvqa_debug_train.pq")
    parser.add_argument("--text-index", default="data/indexes/text_corpus.jsonl")
    parser.add_argument("--image-index", default="data/indexes/image_corpus.jsonl")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--cf-samples", type=int, default=2)
    parser.add_argument("--search-topk", type=int, default=5)
    parser.add_argument("--use-reference", action="store_true", help="Load frozen HF reference policy for logprob.")
    parser.add_argument("--keep-early-answer", action="store_true", help="Do not redact gold answer from step-0 observation.")
    parser.add_argument("--min-future-ig", type=float, default=0.02)
    parser.add_argument("--output", default="results/dagig_offline/dependency_relabel.jsonl")
    parser.add_argument("--selected-output", default="results/dagig_offline/dependency_relabel_selected.jsonl")
    parser.add_argument("--edge-csv", default="paper_artifacts/tables/offline_dependency_edges.csv")
    parser.add_argument("--selected-edge-csv", default="paper_artifacts/tables/offline_dependency_edges_selected.csv")
    parser.add_argument("--summary-csv", default="paper_artifacts/tables/offline_dependency_summary.csv")
    parser.add_argument("--method", default="offline_dependency_relabel")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    policy = None
    if args.use_reference:
        print_gpu_check()
        policy = load_reference_policy_from_config(cfg)

    samples = load_samples(args.samples_jsonl, args.parquet, args.limit)
    trajectories = build_dependency_trajectories(
        samples,
        text_index_path=args.text_index,
        image_index_path=args.image_index,
        search_topk=args.search_topk,
        redact_early_answer=not args.keep_early_answer,
    )

    cf_pool = TypedCounterfactualPool()
    for trajectory in trajectories:
        cf_pool.add_many(trajectory.steps)

    reward_cfg = cfg.get("reward", {})
    reward = DAGIGLiteReward(
        cf_pool=cf_pool,
        local_ig_scorer=LocalIGScorer(
            model=policy,
            cf_pool=cf_pool,
            cf_samples=args.cf_samples,
            dead_zone=float(reward_cfg.get("dead_zone", 0.02)),
            negative_scale=float(reward_cfg.get("negative_scale", 0.25)),
        ),
        future_action_ig_scorer=FutureActionIGScorer(
            model=policy,
            cf_pool=cf_pool,
            cf_samples=args.cf_samples,
            dead_zone=float(reward_cfg.get("dead_zone", 0.02)),
            positive_only=bool(reward_cfg.get("positive_only_future", True)),
        ),
        lambda_dep=float(reward_cfg.get("lambda_dep", 0.5)),
        alpha=float(reward_cfg.get("local_ig_weight", 0.4)),
        beta=float(reward_cfg.get("gate_weight", 0.2)),
        gamma=float(reward_cfg.get("cost_weight", 0.05)),
    )

    relabeled_rows = []
    selected_rows = []
    edge_rows = []
    selected_edge_rows = []
    for trajectory in trajectories:
        output = reward.compute(trajectory)
        reward_by_step = {step.step_id: item for step, item in zip(trajectory.steps, output.step_rewards)}
        diagnostics = output.diagnostics
        edge_row = build_edge_row(trajectory, diagnostics, reward_by_step, args.min_future_ig)
        edge_rows.append(edge_row)
        relabeled_row = {
            "sample_id": trajectory.sample_id,
            "question": trajectory.question,
            "gold_answers": trajectory.gold_answers,
            "final_answer": trajectory.final_answer,
            "final_correct": trajectory.final_correct,
            "method": args.method,
            "selected_for_dependency_training": bool(edge_row["future_credit_eligible"])
            and not bool(edge_row["answer_in_step0_observation"]),
            "steps": [asdict(step) for step in trajectory.steps],
            "step_rewards": [asdict(item) for item in output.step_rewards],
            "token_rewards": output.token_rewards,
            "reward_diagnostics": diagnostics,
            "dependency_edge": edge_row,
        }
        relabeled_rows.append(relabeled_row)
        if relabeled_row["selected_for_dependency_training"]:
            selected_rows.append(relabeled_row)
            selected_edge_rows.append(edge_row)

    write_jsonl(args.output, relabeled_rows)
    write_jsonl(args.selected_output, selected_rows)
    write_csv(args.edge_csv, edge_rows)
    write_csv(args.selected_edge_csv, selected_edge_rows)
    write_csv(args.summary_csv, [summarize_edges(edge_rows, selected_edge_rows, args.method)])
    print(f"saved trajectories={args.output} n={len(relabeled_rows)}")
    print(f"saved selected_trajectories={args.selected_output} n={len(selected_rows)}")
    print(f"saved edge_csv={args.edge_csv}")
    print(f"saved selected_edge_csv={args.selected_edge_csv}")
    print(f"saved summary_csv={args.summary_csv}")


def load_samples(samples_jsonl: str, parquet_path: str, limit: int) -> list[VQASample]:
    samples: list[VQASample] = []
    if samples_jsonl:
        samples = read_samples_jsonl(samples_jsonl)
        if not samples:
            print(f"warning: no usable JSONL samples loaded from {samples_jsonl}")
    else:
        samples = toy_samples()

    samples = [sample for sample in samples if sample.question and sample.gold_answers]
    if samples:
        return samples[:limit]

    parquet = Path(parquet_path)
    if parquet.is_file():
        print(f"loading fallback MMSearch parquet samples from {parquet}")
        return load_mmsearch_parquet_samples(str(parquet), limit)

    raise RuntimeError(
        "No samples available for offline dependency relabel. "
        f"Checked samples_jsonl={samples_jsonl or '<toy default>'} and parquet={parquet_path}. "
        "Run `make prepare_real_data` or `make mmsearch_prepare_fvqa_debug` first."
    )


def load_mmsearch_parquet_samples(path: str, limit: int) -> list[VQASample]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("Reading MMSearch parquet requires pyarrow. Install pyarrow or provide DAGIG_RELABEL_SAMPLES_JSONL.") from exc

    table = pq.read_table(path)
    rows = table.to_pylist()
    samples: list[VQASample] = []
    for idx, row in enumerate(rows):
        sample = mmsearch_row_to_sample(row, idx)
        if sample.question and sample.gold_answers:
            samples.append(sample)
        if len(samples) >= limit:
            break
    if not samples:
        raise RuntimeError(f"No usable samples found in MMSearch parquet: {path}")
    return samples


def mmsearch_row_to_sample(row: dict[str, Any], idx: int) -> VQASample:
    prompt = row.get("prompt") or []
    question = ""
    if isinstance(prompt, list):
        for turn in prompt:
            if isinstance(turn, dict) and str(turn.get("role", "user")).lower() == "user":
                question = str(turn.get("content", "")).strip()
                if question:
                    break
    if not question:
        question = str(row.get("question", row.get("query", ""))).strip()

    reward_model = row.get("reward_model") or {}
    gold_answers: list[str] = []
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        if ground_truth is not None and str(ground_truth).strip():
            gold_answers.append(str(ground_truth).strip())
        gold_answers.extend(parse_candidate_answers(reward_model.get("candidate_answers")))
    if not gold_answers:
        for key in ("answer", "answers", "gold_answers", "ground_truth"):
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                gold_answers.extend(str(item).strip() for item in value if str(item).strip())
            elif str(value).strip():
                gold_answers.append(str(value).strip())
            if gold_answers:
                break

    image_urls = row.get("image_urls", "")
    if isinstance(image_urls, list):
        images = [str(item) for item in image_urls if str(item)]
    elif image_urls:
        images = [str(image_urls)]
    else:
        images = [f"mmsearch_parquet_image_{idx}"]

    extra_info = row.get("extra_info") or {}
    if not isinstance(extra_info, dict):
        extra_info = {}
    sample_id = str(extra_info.get("question_id") or row.get("sample_id") or row.get("id") or f"mmsearch_pq_{idx}")
    return VQASample(
        sample_id=sample_id,
        question=question,
        images=images,
        gold_answers=dedupe(gold_answers),
        metadata={
            "source_dataset": row.get("data_source", "mmsearch_parquet"),
            "source_format": "mmsearch_parquet",
            "needs_search": True,
            "extra_info": extra_info,
        },
    )


def parse_candidate_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str):
        text = str(value).strip()
        return [text] if text else []
    text = value.strip()
    if not text or text == "[]":
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [text]


def dedupe(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        key = value.lower()
        if value and key not in seen:
            out.append(value)
            seen.add(key)
    return out


def build_dependency_trajectories(
    samples: list[VQASample],
    text_index_path: str,
    image_index_path: str,
    search_topk: int,
    redact_early_answer: bool,
) -> list[Trajectory]:
    dispatcher = ToolDispatcher(
        search_topk=search_topk,
        text_index_path=text_index_path,
        image_index_path=image_index_path,
    )
    tokenizer = SimpleTokenizer()
    trajectories: list[Trajectory] = []
    for sample in samples:
        span = 0
        context = f"Question: {sample.question}"
        steps = []
        gold_answers = sample.gold_answers
        gold_answer = gold_answers[0]

        image_action = f"{sample.images[0] if sample.images else sample.sample_id} {sample.question}"
        image_result = dispatcher.run("image_search", image_action, topk=search_topk)
        image_raw = image_result.raw_observation
        if redact_early_answer:
            image_raw = redact_answers(image_raw, gold_answers)
        image_summary = summarize_observation("image_search", image_raw, dispatcher.max_summary_tokens)
        image_step, span, context = _make_step(
            0,
            "image_search",
            image_action,
            image_raw,
            image_summary,
            span,
            tokenizer,
            sample.question,
            context,
            {
                "sample_id": sample.sample_id,
                "success": image_result.success,
                "teacher_generated": True,
                "redacted_gold_answer": redact_early_answer,
            },
        )
        steps.append(image_step)

        anchor = evidence_anchor(image_summary, sample.question)
        text_action = f"{anchor} {sample.question}".strip()
        text_result = dispatcher.run("text_search", text_action, topk=search_topk)
        text_step, span, context = _make_step(
            1,
            "text_search",
            text_action,
            text_result.raw_observation,
            text_result.evidence_summary,
            span,
            tokenizer,
            sample.question,
            context,
            {
                "sample_id": sample.sample_id,
                "success": text_result.success,
                "teacher_generated": True,
                "answer": gold_answer,
            },
        )
        steps.append(text_step)

        stop_result = dispatcher.run("stop", gold_answer)
        stop_step, span, context = _make_step(
            2,
            "stop",
            gold_answer,
            stop_result.raw_observation,
            stop_result.evidence_summary,
            span,
            tokenizer,
            sample.question,
            context,
            {
                "sample_id": sample.sample_id,
                "success": True,
                "teacher_generated": True,
                "answer": gold_answer,
            },
        )
        steps.append(stop_step)

        trajectories.append(
            Trajectory(
                sample_id=sample.sample_id,
                question=sample.question,
                images=sample.images,
                gold_answers=gold_answers,
                steps=steps,
                final_answer=gold_answer,
                final_correct=exact_match(gold_answer, gold_answers),
                full_prompt=sample.question,
                full_response="\n".join(step.action_text for step in steps),
                metadata={
                    "response_token_count": span,
                    "rollout_mode": "teacher_dependency_chain",
                    "tool_free_answers": ["unknown", "uncertain", "unknown"],
                },
            )
        )
    return trajectories


def build_edge_row(
    trajectory: Trajectory,
    diagnostics: dict[str, Any],
    reward_by_step: dict[int, Any],
    min_future_ig: float,
) -> dict[str, Any]:
    local_ig = diagnostics.get("local_ig", {})
    future_ig = diagnostics.get("future_action_ig", {})
    propagated = diagnostics.get("propagated_return", {})
    g0 = float(local_ig.get(0, 0.0))
    g1 = float(local_ig.get(1, 0.0))
    g2 = float(local_ig.get(2, 0.0))
    d01 = float(future_ig.get("0->1", 0.0))
    d12 = float(future_ig.get("1->2", 0.0))
    r0 = float(propagated.get(0, g0))
    r1 = float(propagated.get(1, g1))
    step0 = trajectory.steps[0]
    step1 = trajectory.steps[1]
    answer_in_step0 = any_contains(step0.evidence_summary, trajectory.gold_answers)
    answer_in_step1 = any_contains(step1.evidence_summary, trajectory.gold_answers)
    edge_active = d01 > min_future_ig
    future_credit_eligible = edge_active and max(g1, 0.0) > 0
    return {
        "sample_id": trajectory.sample_id,
        "question": trajectory.question,
        "final_correct": trajectory.final_correct,
        "final_answer": trajectory.final_answer,
        "step0_tool": step0.tool_type,
        "step1_tool": step1.tool_type,
        "g0_local_ig": g0,
        "g1_local_ig": g1,
        "g2_local_ig": g2,
        "d01_future_action_ig": d01,
        "d12_future_action_ig": d12,
        "r0_dagig_return": r0,
        "r1_dagig_return": r1,
        "r0_minus_g0": r0 - g0,
        "answer_in_step0_observation": answer_in_step0,
        "answer_in_step1_observation": answer_in_step1,
        "future_edge_active": edge_active,
        "future_credit_eligible": future_credit_eligible,
        "total_reward_step0": float(reward_by_step[0].total_step_reward),
        "total_reward_step1": float(reward_by_step[1].total_step_reward),
    }


def summarize_edges(rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"method": method, "n": 0}
    return {
        "method": method,
        "n": n,
        "selected_n": len(selected_rows),
        "selected_rate": len(selected_rows) / max(1, n),
        "future_edge_active_rate": mean_bool(rows, "future_edge_active"),
        "future_credit_eligible_rate": mean_bool(rows, "future_credit_eligible"),
        "answer_leak_step0_rate": mean_bool(rows, "answer_in_step0_observation"),
        "answer_supported_step1_rate": mean_bool(rows, "answer_in_step1_observation"),
        "g0_local_ig_mean": mean_float(rows, "g0_local_ig"),
        "g1_local_ig_mean": mean_float(rows, "g1_local_ig"),
        "d01_future_action_ig_mean": mean_float(rows, "d01_future_action_ig"),
        "r0_minus_g0_mean": mean_float(rows, "r0_minus_g0"),
        "selected_d01_future_action_ig_mean": mean_float(selected_rows, "d01_future_action_ig"),
        "selected_r0_minus_g0_mean": mean_float(selected_rows, "r0_minus_g0"),
    }


def evidence_anchor(summary: str, question: str) -> str:
    cleaned = re.sub(r"\b(score|caption|image_id)\s*=\s*", " ", summary, flags=re.IGNORECASE)
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", cleaned)
    question_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", question)}
    keep = []
    for token in tokens:
        if len(token) < 3:
            continue
        if token.lower() in question_tokens:
            continue
        keep.append(token)
        if len(keep) >= 8:
            break
    return " ".join(keep) or question


def redact_answers(value: Any, answers: list[str]) -> Any:
    if isinstance(value, str):
        text = value
        for answer in answers:
            if answer:
                text = re.sub(re.escape(answer), "[REDACTED_ANSWER]", text, flags=re.IGNORECASE)
        return text
    if isinstance(value, dict):
        return {key: redact_answers(item, answers) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_answers(item, answers) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_answers(item, answers) for item in value)
    return value


def any_contains(text: str, answers: list[str]) -> bool:
    text_l = text.lower()
    return any(answer and answer.lower() in text_l for answer in answers)


def mean_float(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0.0)) for row in rows) / max(1, len(rows))


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for row in rows if bool(row.get(key))) / max(1, len(rows))


if __name__ == "__main__":
    main()
