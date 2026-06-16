# Testing and quality gates

The project should move from notebook-based evidence to a professional test suite without losing the notebooks.

## Test pyramid

| Test type | Purpose | First targets |
|---|---|---|
| Unit tests | Validate small functions and schemas. | config loading, metrics, JSONL IO, normalization. |
| Contract tests | Ensure adapters respect interfaces. | LLM provider, retriever, evaluator, exporter. |
| CLI tests | Ensure commands start and fail cleanly. | `ragtree --help`, `ragtree demo`, relation wrappers. |
| Smoke tests | Run minimal end-to-end workflows. | in-memory QA demo, tiny relation extraction demo. |
| Regression tests | Protect benchmark output format. | `pred_relations`, evaluation report schema. |
| Notebook checks | Keep research notebooks runnable when needed. | selected notebooks only, not every PR. |
| Optional integration tests | Use external providers only when credentials are available. | vLLM, Ollama, OpenRouter, vector stores. |

## First must-have tests

1. Package imports without optional dependencies.
2. Core schemas validate and serialize.
3. Relation metrics produce expected TP/FP/FN.
4. Config files load without duplicate-key surprises.
5. `scripts/eval_relations.py` works on a tiny fixture.
6. The minimal Semantic RAG demo runs locally without external services.
7. Current relation experiment wrappers preserve output field names.

## Suggested folder structure

```text
tests/
  unit/
    test_schemas.py
    test_config.py
    test_relation_metrics.py
    test_jsonl_io.py
  contract/
    test_llm_provider_contract.py
    test_retriever_contract.py
    test_task_contract.py
  smoke/
    test_demo_semantic_rag.py
    test_relation_eval_cli.py
  fixtures/
    tiny_documents.jsonl
    tiny_relations_gold.jsonl
    tiny_relations_pred.jsonl
```

## CI quality gates

```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: ruff check ragtree scripts tests
      - run: pytest -q
```

## Rule for optional dependencies

The base package must not require FAISS, Ollama, OpenAI clients, LangGraph, PyTorch, or sentence-transformers just to import. Optional integrations should be imported lazily or installed through extras.
