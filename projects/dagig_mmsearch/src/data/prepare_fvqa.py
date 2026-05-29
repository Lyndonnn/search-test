from __future__ import annotations

from data.dataset_mixer import build_toy_dataset


def main() -> None:
    build_toy_dataset("data/processed/fvqa_toy.jsonl")


if __name__ == "__main__":
    main()

