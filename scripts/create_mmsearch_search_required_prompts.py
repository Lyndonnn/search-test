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

Strict output format:
- First assistant turn must be exactly one search action with reasoning:
  <reason>brief reason for why search is needed</reason><search><img></search>
  or
  <reason>brief reason for the text query</reason><text_search>concise query</text_search>
- Final assistant turn after <information>...</information> must include reasoning and answer:
  <reason>briefly use the returned evidence</reason><answer>final answer</answer>

Do not omit <reason>...</reason>. Do not output a bare <search> or bare <answer>.

Here is the image and the question:
<image> 
"""

IMAGE_SEARCH_REQUIRED_PROMPT = """Answer the user's question based on the provided image.

For this DEBUG exploration run, you must not answer directly in the first assistant turn. You must first invoke image search exactly once by ending your response with <search><img></search>.

Strict output format:
- First assistant turn must be:
  <reason>brief reason for why image search is needed</reason><search><img></search>
- Final assistant turn after <information>...</information> must be:
  <reason>briefly use the returned visual evidence</reason><answer>final answer</answer>

Do not call text search in the first assistant turn. Do not omit <reason>...</reason>. Do not output a bare <search><img></search> or bare <answer>.

Here is the image and the question:
<image> 
"""

TEXT_SEARCH_REQUIRED_PROMPT = """Answer the user's question based on the provided image.

For this DEBUG exploration run, you must not answer directly in the first assistant turn. You must first invoke text search exactly once by ending your response with <text_search>your concise query here</text_search>.

Strict output format:
- First assistant turn must be:
  <reason>brief reason for the text query</reason><text_search>concise query</text_search>
- Final assistant turn after <information>...</information> must be:
  <reason>briefly use the returned evidence</reason><answer>final answer</answer>

Do not call image search in the first assistant turn. Do not omit <reason>...</reason>. Do not output a bare <text_search> or bare <answer>.

Here is the image and the question:
<image> 
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="mmsearch_r1/prompts/round_1_user_prompt_qwenvl_search_required.pkl")
    parser.add_argument("--mode", choices=["search", "image", "text"], default="search")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = {
        "search": SEARCH_REQUIRED_PROMPT,
        "image": IMAGE_SEARCH_REQUIRED_PROMPT,
        "text": TEXT_SEARCH_REQUIRED_PROMPT,
    }[args.mode]
    if "<image>" not in prompt:
        raise RuntimeError("Search-required prompt must keep the <image> placeholder for Qwen-VL.")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(prompt, f)
    print(f"wrote {args.output} mode={args.mode}")


if __name__ == "__main__":
    main()
