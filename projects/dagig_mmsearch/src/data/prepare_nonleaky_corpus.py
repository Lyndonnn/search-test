from __future__ import annotations

import argparse
import re
from typing import Any

from data.dataset_mixer import read_samples_jsonl
from data.schema import VQASample
from utils.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build answer-hidden local retrieval corpora for non-oracle rollout diagnostics.")
    parser.add_argument("--samples-jsonl", default="data/processed/fvqa_train_small.jsonl")
    parser.add_argument("--text-index", default="data/indexes/fvqa_train_nonleaky_text_corpus.jsonl")
    parser.add_argument("--image-index", default="data/indexes/fvqa_train_nonleaky_image_corpus.jsonl")
    parser.add_argument("--include-question-docs", action="store_true")
    parser.add_argument("--max-snippet-chars", type=int, default=360)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_samples_jsonl(args.samples_jsonl)
    text_rows, image_rows = build_nonleaky_indexes(
        samples,
        include_question_docs=args.include_question_docs,
        max_snippet_chars=args.max_snippet_chars,
    )
    write_jsonl(args.text_index, text_rows)
    write_jsonl(args.image_index, image_rows)
    print(f"wrote nonleaky_text_index={args.text_index} n={len(text_rows)}")
    print(f"wrote nonleaky_image_index={args.image_index} n={len(image_rows)}")
    print("nonleaky policy: no answer fields; all known gold answer strings redacted from snippets/captions")


def build_nonleaky_indexes(
    samples: list[VQASample],
    include_question_docs: bool = False,
    max_snippet_chars: int = 360,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_answers = _all_gold_answers(samples)
    text_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        title = _redact_text(str(sample.metadata.get("title") or f"{sample.sample_id} visual evidence"), all_answers)
        evidence = _candidate_evidence(sample, include_question_docs=include_question_docs)
        snippet = _redact_text(evidence, all_answers)
        snippet = _truncate_chars(snippet, max_snippet_chars)
        text_rows.append(
            {
                "doc_id": f"text_{sample.sample_id}",
                "title": title,
                "snippet": snippet,
                "source_sample_id": sample.sample_id,
                "non_leaky": True,
                "contains_gold_answer": _contains_any_answer(snippet, sample.gold_answers),
            }
        )
        image_rows.append(
            {
                "image_id": sample.images[0] if sample.images else f"image_{idx}",
                "title": title,
                "caption": snippet,
                "source_sample_id": sample.sample_id,
                "non_leaky": True,
                "contains_gold_answer": _contains_any_answer(snippet, sample.gold_answers),
            }
        )
    return text_rows, image_rows


def _candidate_evidence(sample: VQASample, include_question_docs: bool) -> str:
    metadata = sample.metadata or {}
    for key in ("evidence", "caption", "context", "passage", "wiki_title"):
        value = metadata.get(key)
        if value and str(value).strip():
            return str(value).strip()
    if include_question_docs:
        return f"Question-like visual context: {sample.question}"
    return (
        "Answer-hidden visual retrieval placeholder. "
        "This document intentionally excludes sample gold answers; replace it with a real corpus for reported experiments."
    )


def _all_gold_answers(samples: list[VQASample]) -> list[str]:
    answers = []
    for sample in samples:
        answers.extend(sample.gold_answers)
    return sorted({answer.strip() for answer in answers if answer and answer.strip()}, key=len, reverse=True)


def _redact_text(text: str, answers: list[str]) -> str:
    redacted = re.sub(r"(?i)\banswer\s*:\s*[^.。\n]+[.。]?", "Answer: [hidden]. ", text)
    redacted = re.sub(r"(?i)\bground[_ ]?truth\s*:\s*[^.。\n]+[.。]?", "Ground truth: [hidden]. ", redacted)
    for answer in answers:
        redacted = re.sub(re.escape(answer), "[hidden]", redacted, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", redacted).strip()


def _contains_any_answer(text: str, answers: list[str]) -> bool:
    text_l = text.lower()
    return any(answer.strip().lower() in text_l for answer in answers if answer and answer.strip())


def _truncate_chars(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


if __name__ == "__main__":
    main()
