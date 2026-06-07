#!/usr/bin/env python3
"""Create debug prompts that force first-turn search for MMSearch-R1 exploration checks."""

from __future__ import annotations

import argparse
import os
import pickle


SEARCH_REQUIRED_PROMPT = """Answer the user's question based on the provided image.

For this DEBUG exploration run, you must not answer directly in the first assistant turn. You must first invoke exactly one search tool:

1. If the image identity, location, object, logo, sign, building, artwork, text, or visual entity is important, call image search by ending your response with <search><img></search>.
2. If the question asks for a factual attribute and you can form a concise query from the question and visible content, call text search by ending your response with <text_search>your concise query here</text_search>.

You must include your reasoning inside <reason>...</reason> before calling a search tool.

After search results are returned inside <information>...</information>, use the returned evidence to answer. When you are ready to answer, wrap your final answer between <answer> and </answer>, without detailed illustrations. For example: <answer>Titanic</answer>.

Here is the image and the question:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="mmsearch_r1/prompts/round_1_user_prompt_qwenvl_search_required.pkl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(SEARCH_REQUIRED_PROMPT, f)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

