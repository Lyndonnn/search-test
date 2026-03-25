import json
import os
import re
from dataclasses import asdict, dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from mmsearch_r1.scripts.inference_torch_demo import generate_response
from mmsearch_r1.utils.tools.image_search import call_image_search
from mmsearch_r1.utils.tools.text_search import call_text_search


MAX_PIXELS = 672 * 672


@dataclass
class GroundedZoomSearchVerifyConfig:
    grid_sizes: Tuple[int, ...] = (1, 2)
    topk_regions: int = 3
    max_zoom_steps: int = 1
    bbox_padding: float = 0.05
    enable_ocr: bool = True
    enable_caption: bool = True
    enable_image_search: bool = True
    enable_text_search: bool = True
    image_search_limit: int = 1
    text_search_limit: int = 1
    region_probe_temperature: float = 0.0


@dataclass
class RegionCandidate:
    region_id: str
    bbox: List[int]
    level: int
    parent_region_id: str
    source: str
    crop_path: str = ""
    relevance: float = 0.0
    needs_zoom: bool = False
    evidence_type: str = "mixed"
    clue: str = ""
    raw_probe: str = ""


@dataclass
class EvidenceItem:
    source: str
    content: str
    confidence: float = 0.0
    region_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if hasattr(value, "item"):
        try:
            return safe_text(value.item())
        except Exception:
            pass
    return str(value).strip()


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if hasattr(value, "item"):
        return to_jsonable(value.item())
    return str(value)


def image_path_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        encoded = f.read()
    import base64

    return f"data:image/png;base64,{base64.b64encode(encoded).decode('utf-8')}"


def clamp_box(box: List[int], image_size: Tuple[int, int]) -> List[int]:
    width, height = image_size
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(1, min(width, int(round(x2))))
    y2 = max(1, min(height, int(round(y2))))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return [x1, y1, x2, y2]


def apply_padding(box: List[int], image_size: Tuple[int, int], ratio: float) -> List[int]:
    if ratio <= 0:
        return clamp_box(box, image_size)
    x1, y1, x2, y2 = box
    pad_w = int(round((x2 - x1) * ratio))
    pad_h = int(round((y2 - y1) * ratio))
    return clamp_box([x1 - pad_w, y1 - pad_h, x2 + pad_w, y2 + pad_h], image_size)


def crop_image(image: Image.Image, box: List[int]) -> Image.Image:
    return image.crop(tuple(box)).convert("RGB")


def generate_grid_boxes(image_size: Tuple[int, int], grid_sizes: Tuple[int, ...]) -> List[Tuple[List[int], str, int]]:
    width, height = image_size
    proposals: List[Tuple[List[int], str, int]] = []
    seen = set()
    for grid in grid_sizes:
        cell_w = width / grid
        cell_h = height / grid
        for row in range(grid):
            for col in range(grid):
                x1 = int(round(col * cell_w))
                y1 = int(round(row * cell_h))
                x2 = int(round((col + 1) * cell_w))
                y2 = int(round((row + 1) * cell_h))
                box = clamp_box([x1, y1, x2, y2], image_size)
                key = tuple(box)
                if key in seen:
                    continue
                seen.add(key)
                proposals.append((box, f"grid_{grid}x{grid}_{row}_{col}", 0))
    return proposals


def subdivide_box(box: List[int]) -> List[List[int]]:
    x1, y1, x2, y2 = box
    mid_x = x1 + (x2 - x1) // 2
    mid_y = y1 + (y2 - y1) // 2
    return [
        [x1, y1, mid_x, mid_y],
        [mid_x, y1, x2, mid_y],
        [x1, mid_y, mid_x, y2],
        [mid_x, mid_y, x2, y2],
    ]


def parse_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}


class GroundedZoomSearchVerifyAgent:
    def __init__(self, model: Any, processor: Any, config: Optional[GroundedZoomSearchVerifyConfig] = None) -> None:
        self.model = model
        self.processor = processor
        self.config = config or GroundedZoomSearchVerifyConfig()
        self.trace_events: List[Dict[str, Any]] = []
        self.local_evidence_cache: List[EvidenceItem] = []
        self.search_results_cache: List[Dict[str, Any]] = []
        self.webpage_evidence_cache: List[Dict[str, Any]] = []
        self.region_history: List[RegionCandidate] = []
        self.tool_budget_usage: Dict[str, int] = {
            "propose_regions": 0,
            "zoom_region": 0,
            "ocr_region": 0,
            "caption_region": 0,
            "crop_image_search": 0,
            "text_search": 0,
            "visit_webpage": 0,
            "summarize_webpage": 0,
            "verify_evidence": 0,
            "answer": 0,
        }

    def _log(self, step: int, action: str, payload: Dict[str, Any]) -> None:
        event = {
            "step": step,
            "action": action,
            "payload": to_jsonable(payload),
            "tool_budget_usage": dict(self.tool_budget_usage),
        }
        self.trace_events.append(event)

    def _ask_on_region(self, image_path: str, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt + "\nRegion image: "},
                    {"type": "image", "image": image_path_to_data_url(image_path), "max_pixels": MAX_PIXELS},
                ],
            }
        ]
        return generate_response(self.model, self.processor, messages)

    def _probe_region(self, question: str, candidate: RegionCandidate) -> RegionCandidate:
        prompt = (
            "You are selecting question-relevant image regions for a grounded zoom-search-verify agent.\n"
            f"Question: {question}\n"
            "Return strict JSON with keys:\n"
            '{"relevance": 0-3, "needs_zoom": true/false, "evidence_type": "text|logo|object|person|document|screen|mixed", "clue": "short clue"}'
        )
        raw = self._ask_on_region(candidate.crop_path, prompt)
        parsed = parse_json_object(raw)
        candidate.relevance = float(parsed.get("relevance", 0.0)) if parsed else 0.0
        candidate.needs_zoom = bool(parsed.get("needs_zoom", False)) if parsed else False
        candidate.evidence_type = safe_text(parsed.get("evidence_type", "mixed")) or "mixed"
        candidate.clue = safe_text(parsed.get("clue", ""))
        candidate.raw_probe = raw
        return candidate

    def _ocr_region(self, question: str, candidate: RegionCandidate) -> EvidenceItem:
        prompt = (
            "Read the region carefully for small text, numbers, logos, labels, or symbols relevant to the question.\n"
            f"Question: {question}\n"
            'Return strict JSON: {"text": "...", "entities": ["..."], "confidence": 0.0}'
        )
        raw = self._ask_on_region(candidate.crop_path, prompt)
        parsed = parse_json_object(raw)
        text = safe_text(parsed.get("text", "")) if parsed else ""
        entities = parsed.get("entities", []) if parsed else []
        joined = " ".join([text] + [safe_text(v) for v in entities if safe_text(v)]).strip()
        return EvidenceItem(
            source="ocr_region",
            content=joined or raw,
            confidence=float(parsed.get("confidence", 0.0)) if parsed else 0.0,
            region_id=candidate.region_id,
            metadata={"raw": raw},
        )

    def _caption_region(self, question: str, candidate: RegionCandidate) -> EvidenceItem:
        prompt = (
            "Describe only the local evidence in this region that is most relevant to the question.\n"
            f"Question: {question}\n"
            'Return strict JSON: {"caption": "...", "entity": "...", "confidence": 0.0}'
        )
        raw = self._ask_on_region(candidate.crop_path, prompt)
        parsed = parse_json_object(raw)
        caption = safe_text(parsed.get("caption", "")) if parsed else ""
        entity = safe_text(parsed.get("entity", "")) if parsed else ""
        joined = " ".join(part for part in [caption, entity] if part).strip()
        return EvidenceItem(
            source="caption_region",
            content=joined or raw,
            confidence=float(parsed.get("confidence", 0.0)) if parsed else 0.0,
            region_id=candidate.region_id,
            metadata={"raw": raw},
        )

    def _build_text_query(self, question: str) -> str:
        clues = [item.content for item in self.local_evidence_cache if item.content]
        clue_text = " | ".join(clues[:3])
        if clue_text:
            return f"{question} Relevant local clues: {clue_text}"
        return question

    def _final_answer(self, question: str) -> Dict[str, Any]:
        evidence_lines = []
        for item in self.local_evidence_cache:
            evidence_lines.append(f"[local:{item.source}] {item.content}")
        for item in self.search_results_cache:
            evidence_lines.append(f"[search:{item['action']}] {safe_text(item.get('returned_text', ''))[:1500]}")
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "No external evidence collected."
        prompt = (
            "You are a grounded zoom-search-verify agent. Answer the question using the provided local and search evidence.\n"
            f"Question: {question}\n"
            f"Evidence:\n{evidence_text}\n"
            'Return strict JSON: {"answer": "...", "evidence_summary": "...", "confidence": 0.0}'
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        raw = generate_response(self.model, self.processor, messages)
        parsed = parse_json_object(raw)
        return {
            "answer": safe_text(parsed.get("answer", "")) if parsed else "",
            "evidence_summary": safe_text(parsed.get("evidence_summary", "")) if parsed else raw,
            "confidence": float(parsed.get("confidence", 0.0)) if parsed else 0.0,
            "raw": raw,
        }

    def _prepare_candidate(self, image: Image.Image, box: List[int], region_id: str, level: int, parent_region_id: str, source: str, workdir: str) -> RegionCandidate:
        crop = crop_image(image, box)
        crop_path = os.path.join(workdir, f"{region_id}.png")
        crop.save(crop_path)
        return RegionCandidate(
            region_id=region_id,
            bbox=box,
            level=level,
            parent_region_id=parent_region_id,
            source=source,
            crop_path=crop_path,
        )

    def run(self, image_path: str, question: str, workdir: str) -> Dict[str, Any]:
        os.makedirs(workdir, exist_ok=True)
        original_image = Image.open(image_path).convert("RGB")
        image_size = original_image.size
        state: Dict[str, Any] = {
            "question": question,
            "original_image": image_path,
            "region_history": [],
            "current_candidate_regions": [],
            "local_evidence_cache": [],
            "search_results_cache": [],
            "webpage_evidence_cache": [],
            "uncertainty_signals": {},
            "tool_budget_usage": self.tool_budget_usage,
        }

        step = 1
        self.tool_budget_usage["propose_regions"] += 1
        proposals = []
        for box, region_id, level in generate_grid_boxes(image_size, self.config.grid_sizes):
            padded = apply_padding(box, image_size, self.config.bbox_padding)
            candidate = self._prepare_candidate(
                original_image,
                padded,
                region_id=region_id,
                level=level,
                parent_region_id="root",
                source="propose_regions",
                workdir=workdir,
            )
            proposals.append(self._probe_region(question, candidate))
        proposals.sort(key=lambda candidate: candidate.relevance, reverse=True)
        top_candidates = proposals[: self.config.topk_regions]
        state["current_candidate_regions"] = [asdict(candidate) for candidate in top_candidates]
        self._log(step, "propose_regions", {"top_candidates": [asdict(candidate) for candidate in top_candidates]})

        selected = top_candidates[0] if top_candidates else self._prepare_candidate(
            original_image,
            [0, 0, image_size[0], image_size[1]],
            region_id="whole_image",
            level=0,
            parent_region_id="root",
            source="fallback",
            workdir=workdir,
        )
        self.region_history.append(selected)

        for zoom_step in range(self.config.max_zoom_steps):
            if not selected.needs_zoom:
                break
            step += 1
            self.tool_budget_usage["zoom_region"] += 1
            child_candidates = []
            for idx, child_box in enumerate(subdivide_box(selected.bbox)):
                padded = apply_padding(child_box, image_size, self.config.bbox_padding)
                child = self._prepare_candidate(
                    original_image,
                    padded,
                    region_id=f"{selected.region_id}_zoom_{zoom_step}_{idx}",
                    level=selected.level + 1,
                    parent_region_id=selected.region_id,
                    source="zoom_region",
                    workdir=workdir,
                )
                child_candidates.append(self._probe_region(question, child))
            child_candidates.sort(key=lambda candidate: candidate.relevance, reverse=True)
            selected = child_candidates[0]
            self.region_history.append(selected)
            self._log(step, "zoom_region", {"selected_region": asdict(selected), "candidates": [asdict(c) for c in child_candidates]})

        state["region_history"] = [asdict(region) for region in self.region_history]

        if self.config.enable_ocr:
            step += 1
            self.tool_budget_usage["ocr_region"] += 1
            ocr_item = self._ocr_region(question, selected)
            self.local_evidence_cache.append(ocr_item)
            self._log(step, "ocr_region", {"region": asdict(selected), "evidence": asdict(ocr_item)})

        if self.config.enable_caption:
            step += 1
            self.tool_budget_usage["caption_region"] += 1
            caption_item = self._caption_region(question, selected)
            self.local_evidence_cache.append(caption_item)
            self._log(step, "caption_region", {"region": asdict(selected), "evidence": asdict(caption_item)})

        if self.config.enable_image_search and self.tool_budget_usage["crop_image_search"] < self.config.image_search_limit:
            step += 1
            self.tool_budget_usage["crop_image_search"] += 1
            returned_text, returned_images, tool_stat = call_image_search(selected.crop_path)
            result = {
                "action": "crop_image_search",
                "query": selected.crop_path,
                "returned_text": returned_text,
                "num_images": len(returned_images),
                "tool_stat": tool_stat,
            }
            self.search_results_cache.append(result)
            self._log(step, "crop_image_search", result)

        if self.config.enable_text_search and self.tool_budget_usage["text_search"] < self.config.text_search_limit:
            step += 1
            self.tool_budget_usage["text_search"] += 1
            text_query = self._build_text_query(question)
            returned_text, tool_stat = call_text_search(text_query)
            result = {
                "action": "text_search",
                "query": text_query,
                "returned_text": returned_text,
                "tool_stat": tool_stat,
            }
            self.search_results_cache.append(result)
            self._log(step, "text_search", result)

        step += 1
        self.tool_budget_usage["verify_evidence"] += 1
        verification = {
            "selected_region_id": selected.region_id,
            "local_evidence_count": len(self.local_evidence_cache),
            "search_result_count": len(self.search_results_cache),
        }
        self._log(step, "verify_evidence", verification)

        step += 1
        self.tool_budget_usage["answer"] += 1
        answer = self._final_answer(question)
        self._log(step, "answer", answer)

        state["local_evidence_cache"] = [asdict(item) for item in self.local_evidence_cache]
        state["search_results_cache"] = self.search_results_cache
        state["webpage_evidence_cache"] = self.webpage_evidence_cache
        state["selected_region"] = asdict(selected)
        state["final_answer"] = answer
        return {
            "state": to_jsonable(state),
            "trace_events": to_jsonable(self.trace_events),
            "final_answer": to_jsonable(answer),
        }
