# ragtree/generation/json_utils.py
"""Robust JSON extraction from LLM output.

LLMs wrap JSON in markdown fences, prepend prose, or emit trailing text.
These helpers recover the first JSON object and normalize relation dicts to
the historical ragtree format used by every benchmark strategy.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

__all__ = ["extract_first_json", "normalize_relations"]

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_first_json(text: str) -> Any | None:
    """Return the first parseable JSON value found in ``text``, else None."""
    if not text:
        return None

    candidates: list[str] = []
    for match in _FENCE_RE.finditer(text):
        candidates.append(match.group(1).strip())
    candidates.append(text.strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass
        start = candidate.find("{")
        while start != -1:
            depth = 0
            for end in range(start, len(candidate)):
                char = candidate[end]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(candidate[start : end + 1])
                        except (json.JSONDecodeError, ValueError):
                            break
            start = candidate.find("{", start + 1)
    return None


def normalize_relations(
    raw: Any, relation_types: Sequence[str] | None = None
) -> dict[str, list[list[str]]]:
    """Normalize a raw relations mapping to ``{TYPE: [[head, tail], ...]}``.

    Mirrors the behavior of the benchmark strategies
    (``BaseRelationStrategy._normalize_relation_dict``): unknown relation
    types are dropped when ``relation_types`` is given, missing ones are
    filled with empty lists, malformed pairs are skipped.
    """
    source = raw if isinstance(raw, dict) else {}
    keys = list(relation_types) if relation_types is not None else list(source.keys())

    normalized: dict[str, list[list[str]]] = {}
    for rtype in keys:
        value = source.get(rtype, [])
        pairs: list[list[str]] = []
        if isinstance(value, list):
            for item in value:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                    and all(isinstance(x, str) for x in item)
                ):
                    pairs.append([item[0], item[1]])
        normalized[rtype] = pairs
    return normalized
