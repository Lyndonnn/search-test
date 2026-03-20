import os
import re
import threading
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import pandas as pd
import requests
from PIL import Image


DEFAULT_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    return [value]


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "item"):
        maybe = value.item()
        if isinstance(maybe, dict):
            return maybe
    return {}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if isinstance(value, list):
        return " ".join(part for part in (_normalize_text(v) for v in value) if part).strip()
    if isinstance(value, dict):
        for key in ("content", "text", "question", "query", "prompt", "value"):
            text = _normalize_text(value.get(key))
            if text:
                return text
        return " ".join(part for part in (_normalize_text(v) for v in value.values()) if part).strip()
    return str(value).strip()


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return []
        return [text]
    if isinstance(value, list):
        return [text for text in (_normalize_text(v) for v in value) if text]
    text = _normalize_text(value)
    return [text] if text else []


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _load_image_from_source(image_source: str) -> Image.Image:
    if image_source.startswith("file://"):
        image_source = image_source[7:]
    if image_source.startswith("http://") or image_source.startswith("https://"):
        headers = dict(DEFAULT_HTTP_HEADERS)
        headers["Referer"] = image_source
        response = requests.get(image_source, timeout=20, headers=headers)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    with open(image_source, "rb") as f:
        return Image.open(BytesIO(f.read())).convert("RGB")


def _image_to_bytes(image: Image.Image, image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def _thumbnail_bytes(image_bytes: bytes, size: int = 224) -> bytes:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((size, size))
    return _image_to_bytes(image, image_format="PNG")


def _compute_dhash(image_bytes: bytes, hash_size: int = 8) -> list[int]:
    image = Image.open(BytesIO(image_bytes)).convert("L").resize((hash_size + 1, hash_size))
    pixels = list(image.getdata())
    diffs: list[int] = []
    width = hash_size + 1
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * width + col]
            right = pixels[row * width + col + 1]
            diffs.append(1 if left > right else 0)
    return diffs


def _compute_histogram(image_bytes: bytes, bins: int = 8) -> list[float]:
    image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((128, 128))
    channels = image.split()
    hist: list[float] = []
    for channel in channels:
        channel_hist = channel.histogram()
        step = 256 // bins
        bucketed = []
        for start in range(0, 256, step):
            bucketed.append(sum(channel_hist[start : start + step]))
        hist.extend(bucketed)
    total = sum(hist) or 1.0
    return [v / total for v in hist]


def _dhash_similarity(lhs: list[int], rhs: list[int]) -> float:
    if not lhs or not rhs or len(lhs) != len(rhs):
        return 0.0
    matches = sum(int(a == b) for a, b in zip(lhs, rhs))
    return matches / len(lhs)


def _hist_similarity(lhs: list[float], rhs: list[float]) -> float:
    if not lhs or not rhs or len(lhs) != len(rhs):
        return 0.0
    return sum(min(a, b) for a, b in zip(lhs, rhs))


@dataclass
class OfflineDoc:
    question_id: str
    question: str
    answer: str
    candidate_answers: list[str]
    category: str
    image_thumb_bytes: bytes
    image_dhash: list[int]
    image_hist: list[float]
    tokens: list[str]
    tf: dict[str, int]
    doc_len: int


class OfflineSearchIndex:
    def __init__(self, parquet_path: str) -> None:
        self.parquet_path = parquet_path
        self.docs = self._load_docs(parquet_path)
        self.doc_count = len(self.docs)
        self.avg_doc_len = sum(doc.doc_len for doc in self.docs) / self.doc_count if self.doc_count else 0.0
        self.doc_freq: dict[str, int] = {}
        for doc in self.docs:
            for token in set(doc.tokens):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def _load_docs(self, parquet_path: str) -> list[OfflineDoc]:
        df = pd.read_parquet(parquet_path)
        docs: list[OfflineDoc] = []
        for row_idx in range(len(df)):
            row = df.iloc[row_idx]
            prompt = _as_sequence(row["prompt"])
            question = prompt[0].get("content", "") if prompt and isinstance(prompt[0], dict) else _normalize_text(prompt)
            reward_model = _as_mapping(row.get("reward_model", {}))
            ground_truth = _normalize_text(reward_model.get("ground_truth"))
            candidate_answers = _normalize_string_list(reward_model.get("candidate_answers"))
            extra_info = _as_mapping(row.get("extra_info", {}))
            question_id = _normalize_text(extra_info.get("question_id")) or str(row_idx)
            category = _normalize_text(extra_info.get("category"))

            images = _as_sequence(row["images"])
            if not images:
                continue
            image_payload = images[0]
            if not isinstance(image_payload, dict) and hasattr(image_payload, "item"):
                image_payload = image_payload.item()
            if not isinstance(image_payload, dict) or "bytes" not in image_payload:
                continue
            image_bytes = image_payload["bytes"]
            thumb_bytes = _thumbnail_bytes(image_bytes)

            doc_text = " ".join(
                part
                for part in [question, ground_truth, " ".join(candidate_answers), category, question_id]
                if part
            )
            tokens = _tokenize(doc_text)
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1

            docs.append(
                OfflineDoc(
                    question_id=question_id,
                    question=question,
                    answer=ground_truth,
                    candidate_answers=candidate_answers,
                    category=category,
                    image_thumb_bytes=thumb_bytes,
                    image_dhash=_compute_dhash(thumb_bytes),
                    image_hist=_compute_histogram(thumb_bytes),
                    tokens=tokens,
                    tf=tf,
                    doc_len=len(tokens),
                )
            )
        return docs

    def search_text(self, query: str, topk: int = 3) -> list[tuple[OfflineDoc, float]]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        k1 = 1.5
        b = 0.75
        scored: list[tuple[OfflineDoc, float]] = []
        for doc in self.docs:
            score = 0.0
            for token in query_tokens:
                tf = doc.tf.get(token, 0)
                if tf == 0:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = max(0.0, ((self.doc_count - df + 0.5) / (df + 0.5)))
                denom = tf + k1 * (1 - b + b * (doc.doc_len / (self.avg_doc_len or 1.0)))
                score += idf * (tf * (k1 + 1)) / (denom or 1.0)
            if score > 0:
                scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topk]

    def search_image(self, image_source: str, topk: int = 3) -> list[tuple[OfflineDoc, float]]:
        query_image = _load_image_from_source(image_source)
        query_thumb = _image_to_bytes(query_image.resize((224, 224)))
        query_dhash = _compute_dhash(query_thumb)
        query_hist = _compute_histogram(query_thumb)

        scored: list[tuple[OfflineDoc, float]] = []
        for doc in self.docs:
            sim = 0.7 * _dhash_similarity(query_dhash, doc.image_dhash) + 0.3 * _hist_similarity(query_hist, doc.image_hist)
            scored.append((doc, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topk]


_INDEX_LOCK = threading.Lock()
_INDEX_CACHE: dict[str, OfflineSearchIndex] = {}


def get_offline_index(parquet_path: Optional[str] = None) -> Optional[OfflineSearchIndex]:
    parquet_path = parquet_path or os.environ.get("MMSEARCH_OFFLINE_PARQUET")
    if not parquet_path or not os.path.exists(parquet_path):
        return None

    with _INDEX_LOCK:
        index = _INDEX_CACHE.get(parquet_path)
        if index is None:
            index = OfflineSearchIndex(parquet_path)
            _INDEX_CACHE[parquet_path] = index
        return index


def format_text_results(results: list[tuple[OfflineDoc, float]]) -> str:
    if not results:
        return "[Text Search Results] No relevant entries were found in the offline FVQA corpus."

    lines = [
        "[Text Search Results] Below are the most relevant entries from the offline FVQA corpus, ranked by lexical relevance:"
    ]
    for rank, (doc, score) in enumerate(results, start=1):
        lines.append(f"{rank}. question_id: {doc.question_id}")
        lines.append(f"   relevance: {score:.4f}")
        lines.append(f"   question: {doc.question}")
        lines.append(f"   answer: {doc.answer}")
        if doc.candidate_answers:
            lines.append(f"   alternate_answers: {', '.join(doc.candidate_answers)}")
        if doc.category:
            lines.append(f"   category: {doc.category}")
    return "\n".join(lines)


def format_image_results(results: list[tuple[OfflineDoc, float]]) -> tuple[str, list[Image.Image], list[str]]:
    if not results:
        return (
            "[Image Search Results] No visually similar entries were found in the offline FVQA corpus.",
            [],
            [],
        )

    lines = [
        "[Image Search Results] Below are visually similar entries from the offline FVQA corpus, ranked by image similarity:"
    ]
    images: list[Image.Image] = []
    titles: list[str] = []
    for rank, (doc, score) in enumerate(results, start=1):
        title = f"Likely answer: {doc.answer}; Related question: {doc.question}"
        titles.append(title)
        lines.append(f"{rank}. question_id: {doc.question_id}")
        lines.append(f"   similarity: {score:.4f}")
        lines.append(f"   related_question: {doc.question}")
        lines.append(f"   likely_answer: {doc.answer}")
        if doc.category:
            lines.append(f"   category: {doc.category}")
        images.append(Image.open(BytesIO(doc.image_thumb_bytes)).convert("RGB"))
    return "\n".join(lines), images, titles
