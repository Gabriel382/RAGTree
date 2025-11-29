# ragtree/processing/rag/base_strategy.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class LLMBackendConfig:
    """
    Generic LLM configuration used across RAG / baseline strategies.
    """
    backend: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    system_prompt: str = ""  # system message content; passed in messages, not to the client


class BaseRelationStrategy(ABC):
    """
    Base class for all relation extraction strategies in ragtree.

    Subclasses must implement `predict_relations`, and can reuse:
      - self.llm_config
      - self.llm_client
      - self._call_llm()
      - self._normalize_relation_dict()
    """

    def __init__(self, llm_config: LLMBackendConfig) -> None:
        self.llm_config = llm_config
        self.llm_client = self._init_llm_client()

    # ------------------------------------------------------------------
    # LLM client initialization
    # ------------------------------------------------------------------
    def _init_llm_client(self) -> Any:
        """
        Initialize the underlying LLM client.

        This assumes a `ragtree.services.llm.get_llm_client` function with
        a signature roughly like:

            get_llm_client(backend: str, model: str, temperature: float, max_tokens: int)

        If your actual function differs, adapt this method accordingly.
        """
        from ragtree.services.llm.llm import get_llm_client  # type: ignore

        client = get_llm_client(
            backend=self.llm_config.backend,
            chat_model=self.llm_config.model,   # or rename to chat_model in LLMBackendConfig
            embed_model=None,
        )
        return client

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the underlying LLM client with a list of messages.

        We assume the client exposes a `chat(messages)` method that returns either:
          - a plain string, or
          - a dict with a 'content' field.

        Adapt this if your LLM interface differs.
        """
        # Common simple pattern: client.chat(messages) -> str
        result = self.llm_client.chat(messages)  # type: ignore[attr-defined]

        if isinstance(result, str):
            return result

        # If it's a dict-like response, try common patterns
        if isinstance(result, dict):
            # e.g. {"content": "..."} or {"choices": [{"message": {"content": "..."}}]}
            if "content" in result and isinstance(result["content"], str):
                return result["content"]

            choices = result.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"]

        # Fallback: string conversion
        return str(result)

    # ------------------------------------------------------------------
    # Helpers for normalizing relation dicts
    # ------------------------------------------------------------------
    def _normalize_relation_dict(
        self,
        raw: Dict[str, Any],
        relation_types: Sequence[str],
    ) -> Dict[str, List[List[str]]]:
        """
        Ensure that the returned dict:
          - has keys exactly from `relation_types` (in the given order),
          - maps to lists of [head_id, tail_id] pairs.

        Unknown relation types from the model are ignored.
        Missing types are filled with empty lists.
        """

        normalized: Dict[str, List[List[str]]] = {}

        for rtype in relation_types:
            value = raw.get(rtype, [])
            pairs: List[List[str]] = []

            if isinstance(value, list):
                for item in value:
                    # item is expected to be ["EVENT_x", "EVENT_y"]
                    if (
                        isinstance(item, list)
                        and len(item) == 2
                        and all(isinstance(x, str) for x in item)
                    ):
                        pairs.append([item[0], item[1]])

            normalized[rtype] = pairs

        return normalized

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    @abstractmethod
    def predict_relations(
        self,
        doc: Dict[str, Any],
        relation_types: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[List[str]]]:
        """
        Given a single document dict, returns a dict:

            {
                "RELATION_TYPE_1": [["EVENT_a", "EVENT_b"], ...],
                "RELATION_TYPE_2": [],
                ...
            }

        `relation_types` is the schema that should be used. If None, the
        subclass decides how to infer it (e.g., from doc["relations"] or a default).
        """
        ...
