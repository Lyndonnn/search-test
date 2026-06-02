from __future__ import annotations

import argparse

from agent.rollout import agentic_search_rollout
from data.dataset_mixer import read_samples_jsonl
from data.schema import toy_samples
from eval.metrics import aggregate_rollouts
from utils.io import write_csv, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight agentic search rollout smoke test.")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--samples-jsonl", default="")
    parser.add_argument("--text-index", default="data/indexes/text_corpus.jsonl")
    parser.add_argument("--image-index", default="data/indexes/image_corpus.jsonl")
    parser.add_argument("--output", default="results/agent_rollout/agentic_rollout_smoke.jsonl")
    parser.add_argument("--table-output", default="paper_artifacts/tables/agentic_rollout_smoke.csv")
    parser.add_argument("--method", default="agentic_search")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_samples_jsonl(args.samples_jsonl) if args.samples_jsonl else toy_samples()
    rows, _ = agentic_search_rollout(
        samples[: args.limit],
        text_index_path=args.text_index,
        image_index_path=args.image_index,
    )
    for row in rows:
        row["method"] = args.method
    write_jsonl(args.output, rows)
    write_csv(args.table_output, [aggregate_rollouts(rows, args.method)])
    print(f"saved {args.output}")
    print(f"saved {args.table_output}")


if __name__ == "__main__":
    main()
