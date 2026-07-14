# ragtree/integrations/llms/mock.py
"""Deterministic mock LLM provider for demos and tests."""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["MockLLMProvider"]


class MockLLMProvider:
    """LLMProvider that needs no network and no extras.

    Parameters
    ----------
    reply:
        Fixed reply for every call.
    reply_fn:
        Callable ``(messages) -> str``; takes precedence over ``reply``.

    With neither given, the provider echoes the last user message.
    """

    def __init__(
        self,
        reply: str | None = None,
        reply_fn: Callable[[list[dict[str, str]]], str] | None = None,
    ) -> None:
        self.reply = reply
        self.reply_fn = reply_fn
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append(list(messages))
        if self.reply_fn is not None:
            return self.reply_fn(list(messages))
        if self.reply is not None:
            return self.reply
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        return f"echo: {last_user}"
