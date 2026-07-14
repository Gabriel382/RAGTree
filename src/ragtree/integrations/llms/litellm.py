# ragtree/integrations/llms/litellm.py
"""LiteLLM provider: one adapter, one hundred model providers."""

from __future__ import annotations

from typing import Any

from ragtree.core.errors import require_extra

__all__ = ["LiteLLMProvider"]


class LiteLLMProvider:
    """LLMProvider over LiteLLM (extra: ``llm-litellm``).

    ``model`` uses LiteLLM naming, e.g. ``openai/gpt-4o-mini`` or
    ``ollama/qwen2.5:7b``. Extra kwargs are forwarded to every completion.
    """

    def __init__(self, model: str, **default_kwargs: Any) -> None:
        require_extra("litellm", "llm-litellm")
        self.model = model
        self.default_kwargs = default_kwargs

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        import litellm

        response = litellm.completion(
            model=self.model,
            messages=list(messages),
            **{**self.default_kwargs, **kwargs},
        )
        choice = response.choices[0]
        message = choice.message if hasattr(choice, "message") else choice["message"]
        content = message.content if hasattr(message, "content") else message["content"]
        return content or ""
