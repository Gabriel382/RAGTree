# ragtree/integrations/llms/ollama.py
"""Ollama provider adapter wrapping the existing research client."""

from __future__ import annotations

from typing import Any

from ragtree.core.errors import require_extra

__all__ = ["OllamaProvider"]


class OllamaProvider:
    """LLMProvider over the local Ollama server (extra: ``llm-ollama``)."""

    def __init__(self, chat_model: str | None = None, embed_model: str | None = None) -> None:
        require_extra("ollama", "llm-ollama")
        from ragtree.services.llm.ollama_client import OllamaClient

        self._client = OllamaClient(chat_model=chat_model, embed_model=embed_model)
        self.model = self._client.chat_model

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return self._client.chat(list(messages), **kwargs)

    def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return [self._client.embed(text) for text in texts]
