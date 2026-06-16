# API design

This document defines the target professional API. It is intentionally broader than the current relation extraction experiments.

## High-level API

```python
from ragtree import RAGTree

pipeline = RAGTree.from_config("examples/configs/semantic_rag.yaml")
result = pipeline.run(task=task, documents=documents)
```

## Core schemas

```python
from pydantic import BaseModel, Field
from typing import Any

class Document(BaseModel):
    id: str
    text: str
    title: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    id: str
    document_id: str
    text: str
    start: int | None = None
    end: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class Evidence(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float | None = None
    retriever: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

## Provider contracts

```python
from typing import Protocol

class LLMProvider(Protocol):
    def complete(self, prompt: str, **kwargs) -> str:
        ...

class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int = 5, **kwargs) -> list[Evidence]:
        ...

class Evaluator(Protocol):
    def evaluate(self, gold, predictions, **kwargs):
        ...
```

## Task contract

```python
class BaseTask(Protocol):
    name: str

    def prepare_query(self, input):
        ...

    def parse_output(self, raw_output: str):
        ...
```

## Backward compatibility

Relation extraction should keep its JSONL fields:

- `document_id`
- `id`
- `type`
- `relations`
- `pred_relations`
- `method`
- `metadata`

This allows the new library layer to coexist with existing benchmark scripts.
