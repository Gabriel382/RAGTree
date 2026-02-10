# ragtree/services/llm.py
from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import os

from ragtree.services.llm.mock_client import BaseLLMClient
from ragtree.services.llm.ollama_client import OllamaClient
from ragtree.services.llm.openrouter_client import OpenRouterClient
from ragtree.services.llm.vllm_client import VLLMClient


# Cache one client per (backend, chat_model, embed_model) triple
_LLM_CLIENT_CACHE: Dict[Tuple[str, Optional[str], Optional[str]], BaseLLMClient] = {}


def _default_backend() -> str:
    return os.getenv("RAGTREE_LLM_BACKEND", "ollama").lower()


def _default_chat_model(backend: str) -> Optional[str]:
    # You can tune these defaults for your setup
    env_val = os.getenv("RAGTREE_LLM_CHAT_MODEL")
    if env_val:
        return env_val

    if backend == "ollama":
        # small but efficient default for local
        return "qwen2.5:3b"
    if backend == "openrouter":
        return "gpt-4.1-mini"

    return None


def _default_embed_model(backend: str) -> Optional[str]:
    env_val = os.getenv("RAGTREE_LLM_EMBED_MODEL")
    if env_val:
        return env_val

    # If you want different defaults per backend, set them here
    if backend == "ollama":
        return None  # or e.g. "nomic-embed-text"
    if backend == "openrouter":
        return None  # or e.g. "text-embedding-3-large"

    return None


def get_llm_client(
    backend: Optional[str] = None,
    chat_model: Optional[str] = None,
    embed_model: Optional[str] = None,
) -> BaseLLMClient:
    """
    Return a (cached) LLM client instance for the requested backend and models.

    Parameters
    ----------
    backend:
        - "ollama"
        - "openrouter"
        - None -> read from env RAGTREE_LLM_BACKEND or default "ollama"

    chat_model:
        - Name/id of the chat model for this backend.
        - None -> read from env RAGTREE_LLM_CHAT_MODEL or backend-specific default.

    embed_model:
        - Name/id of the embedding model for this backend (if used).
        - None -> read from env RAGTREE_LLM_EMBED_MODEL or backend-specific default.

    The function caches one client per (backend, chat_model, embed_model) triple.
    """

    # Resolve backend + models
    backend_name = (backend or _default_backend()).lower()
    chat_name = chat_model or _default_chat_model(backend_name)
    embed_name = embed_model or _default_embed_model(backend_name)

    key = (backend_name, chat_name, embed_name)

    # Reuse from cache if available
    if key in _LLM_CLIENT_CACHE:
        return _LLM_CLIENT_CACHE[key]

    # Instantiate appropriate client
    if backend_name == "ollama":
        client: BaseLLMClient = OllamaClient(
            chat_model=chat_name,
            embed_model=embed_name,
        )
    elif backend_name == "openrouter":
        client = OpenRouterClient(
            chat_model=chat_name,
            embed_model=embed_name,
        )
    
    elif backend_name == "vllm":
        client = VLLMClient(
            chat_model=chat_name,
            embed_model=embed_name,
        )
    else:
        raise ValueError(f"Unknown LLM backend: {backend_name!r}")

    # Tag for debugging / introspection if you like
    client._backend = backend_name          # type: ignore[attr-defined]
    client._chat_model = chat_name          # type: ignore[attr-defined]
    client._embed_model = embed_name        # type: ignore[attr-defined]

    _LLM_CLIENT_CACHE[key] = client
    return client



def embed_text(text: str, backend: Optional[str] = None) -> List[float]:
    """
    Convenience helper: embed text with the selected backend.
    """
    client = get_llm_client(backend=backend)
    return client.embed(text)
