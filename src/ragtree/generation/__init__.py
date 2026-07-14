# ragtree/generation/__init__.py
"""Generation helpers: prompt assembly and robust structured-output parsing."""

from .json_utils import extract_first_json, normalize_relations

__all__ = ["extract_first_json", "normalize_relations"]
