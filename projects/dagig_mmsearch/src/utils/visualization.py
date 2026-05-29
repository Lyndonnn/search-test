from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from utils.io import ensure_dir


_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _fallback_png(path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_bytes(_TINY_PNG)


def plot_hist(values: list[float], path: str | Path, title: str) -> None:
    try:
        ensure_dir(Path(path).parent)
        from PIL import Image, ImageDraw

        values = values or [0.0]
        width, height = 640, 360
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 12), title, fill=(20, 20, 20))
        bins = min(20, max(3, len(values)))
        lo, hi = min(values), max(values)
        if hi == lo:
            hi = lo + 1.0
        counts = [0 for _ in range(bins)]
        for value in values:
            idx = min(bins - 1, int((value - lo) / (hi - lo) * bins))
            counts[idx] += 1
        max_count = max(1, max(counts))
        plot_x, plot_y, plot_w, plot_h = 50, 60, 560, 260
        draw.rectangle((plot_x, plot_y, plot_x + plot_w, plot_y + plot_h), outline=(120, 120, 120))
        bar_w = plot_w / bins
        for idx, count in enumerate(counts):
            x1 = plot_x + int(idx * bar_w)
            x2 = plot_x + int((idx + 1) * bar_w) - 2
            y2 = plot_y + plot_h
            y1 = y2 - int((count / max_count) * plot_h)
            draw.rectangle((x1, y1, x2, y2), fill=(60, 130, 170))
        img.save(path)
    except Exception:
        _fallback_png(path)


def plot_bar(labels: list[str], values: list[float], path: str | Path, title: str) -> None:
    try:
        ensure_dir(Path(path).parent)
        from PIL import Image, ImageDraw

        labels = labels or ["none"]
        values = values or [0.0]
        width, height = 720, 380
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 12), title, fill=(20, 20, 20))
        max_value = max(1e-6, max(abs(v) for v in values))
        plot_x, plot_y, plot_w, plot_h = 60, 60, 620, 240
        draw.rectangle((plot_x, plot_y, plot_x + plot_w, plot_y + plot_h), outline=(120, 120, 120))
        bar_w = plot_w / max(1, len(values))
        zero_y = plot_y + plot_h
        for idx, value in enumerate(values):
            x1 = plot_x + int(idx * bar_w) + 4
            x2 = plot_x + int((idx + 1) * bar_w) - 4
            y1 = zero_y - int((value / max_value) * plot_h)
            draw.rectangle((x1, min(y1, zero_y), x2, max(y1, zero_y)), fill=(130, 90, 60))
            draw.text((x1, plot_y + plot_h + 8), labels[idx][:16], fill=(20, 20, 20))
        img.save(path)
    except Exception:
        _fallback_png(path)


def plot_heatmap(matrix: list[list[float]], path: str | Path, title: str) -> None:
    try:
        ensure_dir(Path(path).parent)
        from PIL import Image, ImageDraw

        matrix = matrix or [[0.0]]
        rows = len(matrix)
        cols = max(1, max(len(row) for row in matrix))
        width, height = 480, 380
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 12), title, fill=(20, 20, 20))
        flat = [value for row in matrix for value in row]
        lo, hi = min(flat), max(flat)
        if hi == lo:
            hi = lo + 1.0
        plot_x, plot_y, plot_w, plot_h = 70, 60, 320, 260
        cell_w = plot_w / cols
        cell_h = plot_h / rows
        for r, row in enumerate(matrix):
            for c in range(cols):
                value = row[c] if c < len(row) else 0.0
                t = (value - lo) / (hi - lo)
                color = (int(40 + 180 * t), int(70 + 120 * (1 - t)), int(120 + 80 * t))
                x1 = plot_x + int(c * cell_w)
                y1 = plot_y + int(r * cell_h)
                x2 = plot_x + int((c + 1) * cell_w)
                y2 = plot_y + int((r + 1) * cell_h)
                draw.rectangle((x1, y1, x2, y2), fill=color, outline=(255, 255, 255))
        img.save(path)
    except Exception:
        _fallback_png(path)


def plot_scatter(xs: list[float], ys: list[float], path: str | Path, title: str, xlabel: str, ylabel: str) -> None:
    try:
        ensure_dir(Path(path).parent)
        from PIL import Image, ImageDraw

        xs = xs or [0.0]
        ys = ys or [0.0]
        width, height = 640, 360
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 12), title, fill=(20, 20, 20))
        plot_x, plot_y, plot_w, plot_h = 60, 60, 520, 240
        draw.rectangle((plot_x, plot_y, plot_x + plot_w, plot_y + plot_h), outline=(120, 120, 120))
        draw.text((plot_x + plot_w // 2 - 40, plot_y + plot_h + 24), xlabel, fill=(20, 20, 20))
        draw.text((10, plot_y + plot_h // 2), ylabel, fill=(20, 20, 20))
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if max_x == min_x:
            max_x = min_x + 1.0
        if max_y == min_y:
            max_y = min_y + 1.0
        for x, y in zip(xs, ys):
            px = plot_x + int((x - min_x) / (max_x - min_x) * plot_w)
            py = plot_y + plot_h - int((y - min_y) / (max_y - min_y) * plot_h)
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(70, 110, 170))
        img.save(path)
    except Exception:
        _fallback_png(path)


def write_case_study_markdown(path: str | Path, trajectory: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    lines = [
        "# DAG-IG Case Study",
        "",
        f"Question: {trajectory.get('question', '')}",
        f"Final answer: {trajectory.get('final_answer', '')}",
        f"Final correct: {trajectory.get('final_correct', False)}",
        "",
    ]
    for step in trajectory.get("steps", []):
        lines.extend(
            [
                f"## Step {step.get('step_id')} - {step.get('tool_type')}",
                f"Action: `{step.get('action_text', '')}`",
                f"Observation summary: {step.get('evidence_summary', '')}",
                f"g_i: {step.get('local_ig', 0.0)}",
                f"d_i_to_j: {step.get('future_action_ig', 0.0)}",
                f"R_i: {step.get('propagated_return', 0.0)}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
