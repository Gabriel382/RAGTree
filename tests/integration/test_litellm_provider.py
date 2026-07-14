"""LiteLLM adapter: construction + protocol conformance (no network call)."""

import pytest

pytest.importorskip("litellm")

from ragtree.core.protocols import LLMProvider
from ragtree.integrations.llms import LiteLLMProvider

pytestmark = [pytest.mark.integration]


def test_satisfies_protocol_without_calling_out():
    provider = LiteLLMProvider(model="openai/gpt-4o-mini", temperature=0.0)
    assert isinstance(provider, LLMProvider)
    assert provider.model == "openai/gpt-4o-mini"
    assert provider.default_kwargs == {"temperature": 0.0}
