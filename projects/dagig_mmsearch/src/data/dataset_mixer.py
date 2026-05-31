from __future__ import annotations

from pathlib import Path

from data.schema import VQASample, sample_to_dict, toy_samples
from utils.io import read_jsonl, write_jsonl


def build_toy_dataset(output_path: str = "data/processed/toy_vqa.jsonl", limit: int | None = None) -> list[VQASample]:
    samples = toy_samples()
    if limit is not None:
        samples = samples[:limit]
    write_jsonl(output_path, [sample_to_dict(sample) for sample in samples])
    return samples


def build_toy_indexes(
    text_path: str = "data/indexes/text_corpus.jsonl",
    image_path: str = "data/indexes/image_corpus.jsonl",
) -> None:
    text_rows = [
        {"title": "Eiffel Tower location", "snippet": "The Eiffel Tower is located in Paris, France.", "answer": "Paris"},
        {"title": "Mona Lisa museum", "snippet": "The Mona Lisa is displayed in the Louvre Museum.", "answer": "Louvre Museum"},
        {"title": "Golden Gate Bridge city", "snippet": "The Golden Gate Bridge is in San Francisco.", "answer": "San Francisco"},
        {"title": "Statue of Liberty city", "snippet": "The Statue of Liberty is in New York City.", "answer": "New York City"},
        {"title": "Colosseum city", "snippet": "The Colosseum is in Rome.", "answer": "Rome"},
        {"title": "Sydney Opera House", "snippet": "The Sydney Opera House is located in Sydney.", "answer": "Sydney"},
        {"title": "Taj Mahal home city", "snippet": "The Taj Mahal is in Agra, India.", "answer": "Agra"},
        {"title": "Pyramids of Giza", "snippet": "The Pyramids of Giza are near Cairo.", "answer": "Cairo"},
    ]
    image_rows = [
        {"image_id": "img_eiffel", "title": "Eiffel Tower", "caption": "Paris landmark tower.", "answer": "Paris"},
        {"image_id": "img_louvre", "title": "Mona Lisa Louvre", "caption": "Museum context points to the Louvre Museum.", "answer": "Louvre Museum"},
        {"image_id": "img_golden_gate", "title": "Golden Gate Bridge", "caption": "San Francisco red suspension bridge.", "answer": "San Francisco"},
        {"image_id": "img_liberty", "title": "Statue of Liberty", "caption": "New York City harbor monument.", "answer": "New York City"},
        {"image_id": "img_colosseum", "title": "Colosseum", "caption": "Ancient amphitheater in Rome.", "answer": "Rome"},
        {"image_id": "img_sydney", "title": "Sydney Opera House", "caption": "Harbor performing arts venue in Sydney.", "answer": "Sydney"},
        {"image_id": "img_taj", "title": "Taj Mahal", "caption": "White marble mausoleum in Agra.", "answer": "Agra"},
        {"image_id": "img_giza", "title": "Pyramids of Giza", "caption": "Ancient pyramids near Cairo.", "answer": "Cairo"},
    ]
    write_jsonl(text_path, text_rows)
    write_jsonl(image_path, image_rows)
    Path("data/cache").mkdir(parents=True, exist_ok=True)


def build_indexes_from_samples(
    samples: list[VQASample],
    text_path: str = "data/indexes/sample_text_corpus.jsonl",
    image_path: str = "data/indexes/sample_image_corpus.jsonl",
) -> None:
    text_rows = []
    image_rows = []
    for idx, sample in enumerate(samples):
        answer = sample.gold_answers[0] if sample.gold_answers else ""
        title = sample.metadata.get("title") or f"{sample.sample_id} evidence"
        snippet = sample.metadata.get("evidence") or (
            f"Question: {sample.question} Answer: {answer}. "
            f"This local diagnostic evidence is generated from the sample gold answer."
        )
        text_rows.append(
            {
                "title": str(title),
                "snippet": str(snippet),
                "answer": answer,
                "sample_id": sample.sample_id,
            }
        )
        image_rows.append(
            {
                "image_id": sample.images[0] if sample.images else f"sample_image_{idx}",
                "title": str(title),
                "caption": str(snippet),
                "answer": answer,
                "sample_id": sample.sample_id,
            }
        )
    write_jsonl(text_path, text_rows)
    write_jsonl(image_path, image_rows)


def read_samples_jsonl(path: str) -> list[VQASample]:
    from data.schema import sample_from_dict

    return [sample_from_dict(row) for row in read_jsonl(path)]


if __name__ == "__main__":
    build_toy_dataset()
    build_toy_indexes()
