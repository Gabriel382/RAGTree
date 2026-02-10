# ragtree/processing/llm/ollama_client.py
from __future__ import annotations

from typing import List, Sequence, Dict, Any, Optional

import ollama  # pip install ollama

from ragtree.services.llm.mock_client import BaseLLMClient


class OllamaClient(BaseLLMClient):
    """
    Simple Ollama-based client implementing BaseLLMClient.

    You can configure models via env vars later if you want:
      - OLLAMA_CHAT_MODEL
      - OLLAMA_EMBED_MODEL
    """

    def __init__(
        self,
        chat_model: Optional[str] = None,
        embed_model: Optional[str] = None,
    ) -> None:
        import os

        self.chat_model = chat_model or os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
        self.embed_model = embed_model or os.getenv(
            "OLLAMA_EMBED_MODEL",
            "nomic-embed-text",
        )

    def embed(self, text: str) -> List[float]:
        resp = ollama.embeddings(model=self.embed_model, prompt=text)
        return resp["embedding"]

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        For now we just pass the last user message as prompt.
        If you want full multi-message support, you can concatenate.
        """
        if not messages:
            raise ValueError("OllamaClient.chat: messages must not be empty")

        # simple strategy: join all messages into a single prompt
        prompt = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages if "content" in m
        )

        resp = ollama.chat(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return resp["message"]["content"]
