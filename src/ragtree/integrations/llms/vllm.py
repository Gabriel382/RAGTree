# ragtree/integrations/llms/vllm.py
"""vLLM / OpenAI-compatible HTTP provider (no extra needed: uses requests)."""

from __future__ import annotations

from typing import Any

__all__ = ["VLLMProvider"]


class VLLMProvider:
    """LLMProvider over a vLLM (or any OpenAI-compatible) HTTP endpoint.

    Configuration via constructor or env: VLLM_BASE_URL, VLLM_API_KEY,
    VLLM_CHAT_MODEL, VLLM_EMBED_MODEL.
    """

    def __init__(self, chat_model: str | None = None, embed_model: str | None = None) -> None:
        from ragtree.services.llm.vllm_client import VLLMClient

        self._client = VLLMClient(chat_model=chat_model, embed_model=embed_model)
        self.model = self._client.chat_model

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return self._client.chat(list(messages), **kwargs)

    def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return [self._client.embed(text) for text in texts]
