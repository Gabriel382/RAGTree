# ragtree/processing/llm/openrouter_client.py
from __future__ import annotations

from typing import List, Sequence, Dict, Any, Optional

import os

from openai import OpenAI  # pip install openai

from ragtree.services.llm.mock_client import BaseLLMClient


class OpenRouterClient(BaseLLMClient):
    """
    OpenRouter-based LLM client implementing BaseLLMClient.

    Requires:
      - env OPENROUTER_API_KEY
    """

    def __init__(
        self,
        chat_model: Optional[str] = None,
        embed_model: Optional[str] = None,
    ) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        self.chat_model = chat_model or os.getenv(
            "OPENROUTER_CHAT_MODEL",
            "gpt-4o-mini",  # example, adjust to what you use
        )
        self.embed_model = embed_model or os.getenv(
            "OPENROUTER_EMBED_MODEL",
            "text-embedding-3-large",  # example
        )

    def embed(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(
            model=self.embed_model,
            input=text,
        )
        return list(resp.data[0].embedding)

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        chat_resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=list(messages),
            **kwargs,
        )
        return chat_resp.choices[0].message.content or ""
