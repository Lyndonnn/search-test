from __future__ import annotations

import argparse

from agent.policy_wrapper import PolicyWrapper, SimpleTokenizer
from agent.rollout import agentic_search_rollout
from data.dataset_mixer import read_samples_jsonl
from data.schema import toy_samples
from eval.metrics import aggregate_rollouts
from train.trainer_utils import load_config
from utils.gpu_check import main as print_gpu_check
from utils.hf_reference import load_reference_policy_from_config
from utils.io import write_csv, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen/HF-generated one-action multimodal search rollout.")
    parser.add_argument("--config", default="projects/dagig_mmsearch/configs/dagig_lite_qwen25vl_3b_a800.yaml")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--samples-jsonl", default="")
    parser.add_argument("--text-index", default="data/indexes/text_corpus.jsonl")
    parser.add_argument("--image-index", default="data/indexes/image_corpus.jsonl")
    parser.add_argument("--output", default="results/model_agent/model_agent_rollout.jsonl")
    parser.add_argument("--table-output", default="paper_artifacts/tables/model_agent_rollout.csv")
    parser.add_argument("--method", default="model_agent_qwen25vl3b")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--scripted-direct-stop", action="store_true")
    parser.add_argument("--force-search-when-needed", action="store_true")
    parser.add_argument("--fallback-on-invalid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    print_gpu_check()
    hf_policy = load_reference_policy_from_config(cfg)
    policy = PolicyWrapper(model=hf_policy, tokenizer=SimpleTokenizer())
    samples = read_samples_jsonl(args.samples_jsonl) if args.samples_jsonl else toy_samples()
    rows, _ = agentic_search_rollout(
        samples[: args.limit],
        text_index_path=args.text_index,
        image_index_path=args.image_index,
        policy=policy,
        scripted_direct_stop=args.scripted_direct_stop,
        force_search_when_needed=args.force_search_when_needed,
        fallback_on_invalid=args.fallback_on_invalid,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    for row in rows:
        row["method"] = args.method
    write_jsonl(args.output, rows)
    write_csv(args.table_output, [aggregate_rollouts(rows, args.method)])
    print(f"saved {args.output}")
    print(f"saved {args.table_output}")


if __name__ == "__main__":
    main()
