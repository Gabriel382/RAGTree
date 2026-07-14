# ragtree/services/llm/vllm_client.py
from __future__ import annotations

import os
import requests
from typing import List, Sequence, Dict, Any, Optional

from ragtree.services.llm.mock_client import BaseLLMClient


class VLLMClient(BaseLLMClient):
    """
    vLLM HTTP backend using /v1/chat/completions and optionally /v1/embeddings.

    Required env:
      - VLLM_BASE_URL   (e.g. "http://localhost:8000")
      - VLLM_API_KEY (optional if your server doesn't require real auth)
    """

    def __init__(
        self,
        chat_model: Optional[str] = None,
        embed_model: Optional[str] = None,
    ) -> None:
        self.base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
        self.api_key = os.getenv("VLLM_API_KEY", "dummy")

        self.chat_model = chat_model or os.getenv("VLLM_CHAT_MODEL", "openai/gpt-oss-20b")
        self.embed_model = embed_model or os.getenv("VLLM_EMBED_MODEL", "openai/text-embedding-3-large")

        self._chat_url = f"{self.base_url}/v1/chat/completions"
        self._embed_url = f"{self.base_url}/v1/embeddings"

        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def embed(self, text: str) -> List[float]:
        payload = {
            "model": self.embed_model,
            "input": text,
        }
        resp = requests.post(self._embed_url, json=payload, headers=self._headers)
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        payload = {
            "model": self.chat_model,
            "messages": list(messages),
        }
        payload.update(kwargs)

        resp = requests.post(self._chat_url, json=payload, headers=self._headers)
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            or ""
        )
