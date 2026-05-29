from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.io import ensure_dir, read_json, write_json


class JsonToolCache:
    def __init__(self, cache_dir: str = "data/cache/tools") -> None:
        self.cache_dir = Path(cache_dir)
        ensure_dir(self.cache_dir)

    def _path(self, key: str) -> Path:
        safe = str(abs(hash(key)))
        return self.cache_dir / f"{safe}.json"

    def get(self, key: str) -> Any:
        return read_json(self._path(key), default=None)

    def set(self, key: str, value: Any) -> None:
        write_json(self._path(key), value)

