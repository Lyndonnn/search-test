from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


@dataclass
class LogProbCache:
    values: dict[tuple[str, str], float] = field(default_factory=dict)

    def get(self, context: str, target: str) -> float | None:
        return self.values.get((context, target))

    def set(self, context: str, target: str, value: float) -> None:
        self.values[(context, target)] = value


class FrozenLogProbScorer:
    """Reference-policy logprob wrapper with a deterministic CPU fallback.

    If a model exposes `score_text(context, target)`, that API is used. Otherwise
    a lexical scorer provides stable smoke-test behavior without model downloads.
    """

    def __init__(self, model: Any = None, tokenizer: Any = None, length_norm: bool = True) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.length_norm = length_norm
        self.cache = LogProbCache()

    def score(self, context: str, target: str) -> float:
        cached = self.cache.get(context, target)
        if cached is not None:
            return cached

        if self.model is not None and hasattr(self.model, "score_text"):
            value = float(self.model.score_text(context, target))
        elif self.model is not None and self.tokenizer is not None:
            value = self._hf_logprob(context, target)
        else:
            value = self._lexical_logprob(context, target)

        self.cache.set(context, target, value)
        return value

    def batch_score(self, contexts: list[str], targets: list[str]) -> list[float]:
        return [self.score(context, target) for context, target in zip(contexts, targets)]

    def _lexical_logprob(self, context: str, target: str) -> float:
        context_tokens = set(_tokens(context))
        target_tokens = _tokens(target)
        if not target_tokens:
            return 0.0
        matches = sum(1 for token in target_tokens if token in context_tokens)
        coverage = matches / max(1, len(target_tokens))
        length_penalty = 0.02 * max(0, len(target_tokens) - matches)
        return math.log(0.05 + 0.9 * coverage) - length_penalty

    def _hf_logprob(self, context: str, target: str) -> float:
        try:
            import torch

            with torch.no_grad():
                context_ids = self.tokenizer(context, return_tensors="pt").input_ids
                target_ids = self.tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids
                input_ids = torch.cat([context_ids, target_ids], dim=1)
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits[:, :-1, :]
                labels = input_ids[:, 1:]
                log_probs = torch.log_softmax(logits, dim=-1)
                start = context_ids.shape[1] - 1
                end = start + target_ids.shape[1]
                token_log_probs = log_probs[:, start:end, :].gather(2, labels[:, start:end].unsqueeze(-1)).squeeze(-1)
                value = float(token_log_probs.sum().item())
                if self.length_norm and target_ids.shape[1] > 0:
                    value /= float(target_ids.shape[1])
                return value
        except Exception:
            return self._lexical_logprob(context, target)


class ToyLogProbModel:
    def __init__(self, boosts: dict[str, float] | None = None) -> None:
        self.boosts = boosts or {}

    def score_text(self, context: str, target: str) -> float:
        context_l = context.lower()
        target_l = target.lower()
        score = -1.0
        for key, value in self.boosts.items():
            if key.lower() in context_l or key.lower() in target_l:
                score += value
        target_tokens = _tokens(target)
        if target_tokens:
            context_tokens = set(_tokens(context))
            score += sum(token in context_tokens for token in target_tokens) / len(target_tokens)
        return float(score)

