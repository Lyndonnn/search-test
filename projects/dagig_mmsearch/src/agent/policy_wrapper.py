from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PolicyOutput:
    text: str
    tokens: list[int]
    metadata: dict[str, Any]


class SimpleTokenizer:
    def encode(self, text: str) -> list[int]:
        return [abs(hash(tok)) % 32000 for tok in text.split()]

    def decode(self, tokens: list[int]) -> str:
        return " ".join(str(token) for token in tokens)


class PolicyWrapper:
    def __init__(self, model: Any = None, tokenizer: Any = None) -> None:
        self.model = model
        self.tokenizer = tokenizer or SimpleTokenizer()

    def generate(self, prompt: str, **_: Any) -> PolicyOutput:
        if self.model is not None and hasattr(self.model, "generate_text"):
            text = str(self.model.generate_text(prompt))
        else:
            text = self._scripted_response(prompt)
        return PolicyOutput(text=text, tokens=self.tokenizer.encode(text), metadata={"scripted": self.model is None})

    def _scripted_response(self, prompt: str) -> str:
        prompt_l = prompt.lower()
        if "eiffel" in prompt_l:
            return '{"tool": "text_search", "action": "Eiffel Tower location"}'
        if "mona lisa" in prompt_l:
            return '{"tool": "text_search", "action": "Mona Lisa museum"}'
        if "golden gate" in prompt_l:
            return '{"tool": "image_search", "action": "Golden Gate Bridge"}'
        return '{"tool": "stop", "action": "unknown"}'

