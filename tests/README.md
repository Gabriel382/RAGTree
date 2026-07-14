# RAGTree test suite

Layered per the BYOS design document (section 10):

| Folder | Purpose | Needs extras? |
|---|---|---|
| `unit/` | Pure core logic: schemas, config, registry, errors, CLI. | No |
| `contract/` | Protocol conformance. `bases.py` holds reusable contract test classes; adapters subclass them and provide a fixture. | No (fakes) |
| `integration/` | Real optional stacks (Chroma, Qdrant, Neo4j, FastAPI, ...). Marked, skipped without extras. | Yes |
| `e2e/` | Tiny full pipelines over fixture data. | Mock LLM only |
| `regression/` | Protect current experiment output formats (`pred_relations`). | No |
| `fixtures/` | Tiny committed datasets (exempt from the repo-wide `*.json`/`*.jsonl` gitignore). | — |

Fast local run: `pytest tests/unit tests/contract`
