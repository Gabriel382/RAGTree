# ragtree/integrations/llms/openrouter.py
"""OpenRouter provider adapter wrapping the existing research client."""

from __future__ import annotations

from typing import Any

from ragtree.core.errors import require_extra

__all__ = ["OpenRouterProvider"]


class OpenRouterProvider:
    """LLMProvider over OpenRouter (extra: ``llm-openai``; needs OPENROUTER_API_KEY)."""

    def __init__(self, chat_model: str | None = None, embed_model: str | None = None) -> None:
        require_extra("openai", "llm-openai")
        from ragtree.services.llm.openrouter_client import OpenRouterClient

        self._client = OpenRouterClient(chat_model=chat_model, embed_model=embed_model)
        self.model = getattr(self._client, "chat_model", chat_model)

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return self._client.chat(list(messages), **kwargs)

    def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return [self._client.embed(text) for text in texts]
