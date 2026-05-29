from __future__ import annotations


def find_action_span(response_tokens: list[int], action_tokens: list[int]) -> tuple[int, int]:
    if not action_tokens:
        return (0, 0)
    limit = len(response_tokens) - len(action_tokens) + 1
    for start in range(max(0, limit)):
        if response_tokens[start : start + len(action_tokens)] == action_tokens:
            return (start, start + len(action_tokens))
    return (0, len(action_tokens))


def safe_span_length(span: tuple[int, int], fallback_tokens: list[int] | None = None) -> int:
    start, end = span
    length = max(0, end - start)
    if length == 0 and fallback_tokens:
        return len(fallback_tokens)
    return max(1, length)

