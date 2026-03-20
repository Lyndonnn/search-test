import os
from io import BytesIO
from typing import Any, Optional

import requests
from PIL import Image


SERPAPI_ENDPOINT = "https://serpapi.com/search"


def get_serpapi_key() -> str:
    return os.environ.get("SERPAPI_API_KEY", "").strip()


def has_serpapi() -> bool:
    return bool(get_serpapi_key())


def _request_serpapi(params: dict[str, Any]) -> dict[str, Any]:
    api_key = get_serpapi_key()
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY is not set")

    query = dict(params)
    query["api_key"] = api_key
    response = requests.get(SERPAPI_ENDPOINT, params=query, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(payload["error"])
    return payload


def _download_image(url: str) -> Optional[Image.Image]:
    if not url:
        return None
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        return None


def search_google(text_query: str, num_results: int = 3, hl: str = "en", gl: str = "us") -> tuple[str, dict[str, Any]]:
    payload = _request_serpapi(
        {
            "engine": "google",
            "q": text_query,
            "num": num_results,
            "hl": hl,
            "gl": gl,
        }
    )

    lines = [
        "[Text Search Results] Below are the text summaries of the most relevant webpages related to your query, ranked in descending order of relevance:"
    ]

    count = 0
    answer_box = payload.get("answer_box") or {}
    if answer_box:
        snippet = answer_box.get("snippet") or answer_box.get("answer") or answer_box.get("title") or ""
        if snippet:
            lines.append(f"1. {snippet}")
            count += 1

    knowledge_graph = payload.get("knowledge_graph") or {}
    if knowledge_graph and count < num_results:
        title = knowledge_graph.get("title") or ""
        description = knowledge_graph.get("description") or ""
        kg_text = " - ".join(part for part in [title, description] if part)
        if kg_text:
            lines.append(f"{count + 1}. {kg_text}")
            count += 1

    for result in payload.get("organic_results", []):
        if count >= num_results:
            break
        title = result.get("title") or ""
        snippet = result.get("snippet") or ""
        source = result.get("source") or result.get("displayed_link") or result.get("link") or ""
        text = " | ".join(part for part in [title, snippet, source] if part)
        if text:
            lines.append(f"{count + 1}. {text}")
            count += 1

    if count == 0:
        lines.append("1. No relevant results returned by SerpApi Google Search.")

    return "\n".join(lines), {"success": True, "num_results": count, "backend": "serpapi_google"}


def search_google_lens(image_url: str, num_results: int = 3, hl: str = "en", country: str = "us") -> tuple[str, list[Image.Image], dict[str, Any]]:
    if not image_url.startswith("http://") and not image_url.startswith("https://"):
        raise RuntimeError(
            "SerpApi Google Lens requires a public image URL. Local file paths are not directly supported."
        )

    payload = _request_serpapi(
        {
            "engine": "google_lens",
            "url": image_url,
            "type": "visual_matches",
            "hl": hl,
            "country": country,
        }
    )

    lines = [
        "[Image Search Results] The result of the image search consists of web page information related to the image from the user's original question. Each result includes the main image from the web page and its title, ranked in descending order of search relevance, as demonstrated below:"
    ]
    images: list[Image.Image] = []
    titles: list[str] = []
    count = 0

    for result in payload.get("visual_matches", []):
        if count >= num_results:
            break
        title = result.get("title") or result.get("source") or "Untitled result"
        link = result.get("link") or ""
        source = result.get("source") or ""
        snippet = result.get("price", {}).get("value") if isinstance(result.get("price"), dict) else ""
        lines.append(f"{count + 1}. title: {title}")
        if source:
            lines.append(f"   source: {source}")
        if link:
            lines.append(f"   link: {link}")
        if snippet:
            lines.append(f"   extra: {snippet}")
        titles.append(title)
        image = _download_image(result.get("thumbnail") or result.get("image") or "")
        if image is not None:
            images.append(image)
        count += 1

    if count == 0:
        lines.append("1. No visual matches returned by SerpApi Google Lens.")

    return "\n".join(lines), images, {"success": True, "num_images": len(images), "backend": "serpapi_google_lens", "titles": titles}
