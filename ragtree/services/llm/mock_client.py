# ragtree/processing/llm/mock_client.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Dict, Any


class BaseLLMClient(ABC):
    """
    Abstract interface for all LLM backends (Ollama, OpenRouter, etc.).

    Implementations MUST at least provide:
      - embed(text) -> List[float]
      - chat(messages) -> str  (simple text response)
    """

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Return a single embedding vector for `text`.
        """
        raise NotImplementedError

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        Simple chat completion. Messages follow the OpenAI-style format:
          [{"role": "system"|"user"|"assistant", "content": "..."}]

        Returns the assistant's message content as a string.
        """
        raise NotImplementedError
