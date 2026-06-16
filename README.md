# RAGTree

<div align="center">
  <img src="docs/assets/ragtree-icon.png" alt="RAGTree icon" width="135" />

  <h3>Bring-your-own-stack Semantic RAG framework</h3>

  <p>
    Build, evaluate, and integrate grounded RAG pipelines over complex documents.
  </p>

  <p>
    <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue">
    <img alt="Semantic RAG" src="https://img.shields.io/badge/focus-Semantic%20RAG-0f766e">
    <img alt="BYOS" src="https://img.shields.io/badge/design-bring%20your%20own%20stack-073763">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  </p>
</div>

---

## Overview

**RAGTree** is a bring-your-own-stack framework for Semantic RAG pipelines.

The project is designed around a simple rule:

> The core defines contracts. Integrations implement contracts. Tests verify contracts.

RAGTree started as a research workbench for relation extraction, ontology-guided retrieval, KG-RAG, agentic RAG, evaluation, runtime analysis and CO2 estimation. The next architecture keeps those experiments runnable while moving the repository toward an installable library with a clear source layout, optional addons and explicit test layers.

The goal is not to force users into one stack. Users should be able to bring their own:

- LLM provider;
- embedding model;
- vector database;
- retriever;
- ontology or knowledge graph;
- evaluator;
- exporter;
- API or UI layer.

---

## Why this repository is structured this way

RAG projects become difficult to maintain when research scripts, provider SDKs, APIs, dashboards and tests live in the same layer.

RAGTree separates them:

| Layer | Role |
|---|---|
| `src/ragtree/core/` | Stable schemas, protocols, config, registry, errors and pipeline contracts. |
| `src/ragtree/tasks/` | Task definitions such as question answering, relation extraction, summarization and verification. |
| `src/ragtree/retrieval/` | Retrieval orchestration independent of a specific vector database. |
| `src/ragtree/generation/` | Prompting, structured generation, JSON repair and generator contracts. |
| `src/ragtree/evaluation/` | Metrics, reports, faithfulness, runtime and CO2 analysis. |
| `src/ragtree/integrations/` | Optional adapters for external libraries and services. |
| `src/ragtree/apps/` | Optional FastAPI and Streamlit surfaces. |
| `tests/` | Unit, contract, integration, end-to-end and regression tests. |
| `experiments/` | Reproducible benchmark workflows and preserved research scripts. |

---

## Target repository layout

```text
RAGTree/
├── src/
│   └── ragtree/
│       ├── core/
│       │   ├── schemas.py
│       │   ├── protocols.py
│       │   ├── config.py
│       │   ├── pipeline.py
│       │   ├── registry.py
│       │   └── errors.py
│       ├── tasks/
│       │   ├── base.py
│       │   ├── question_answering.py
│       │   ├── relation_extraction.py
│       │   ├── summarization.py
│       │   ├── claim_verification.py
│       │   └── graph_construction.py
│       ├── retrieval/
│       │   ├── base.py
│       │   ├── dense.py
│       │   ├── hybrid.py
│       │   ├── ontology_guided.py
│       │   └── kg_guided.py
│       ├── generation/
│       │   ├── base.py
│       │   ├── prompts.py
│       │   ├── structured.py
│       │   └── repair.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── relation_metrics.py
│       │   ├── faithfulness.py
│       │   └── reports.py
│       ├── integrations/
│       │   ├── llms/
│       │   ├── vectorstores/
│       │   ├── graphstores/
│       │   ├── ontologies/
│       │   └── agents/
│       ├── apps/
│       │   ├── api/
│       │   └── streamlit/
│       └── cli/
│           └── main.py
├── tests/
│   ├── unit/
│   ├── contract/
│   ├── integration/
│   ├── e2e/
│   ├── regression/
│   └── fixtures/
├── examples/
├── experiments/
├── scripts/
├── docs/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Core protocol rule

Core modules must not import optional integration SDKs directly.

This means:

```text
core/ does not import Chroma, Qdrant, Neo4j, FastAPI, Streamlit, LangGraph or LiteLLM.
```

Instead, the core defines protocols:

```python
from typing import Protocol

class LLMProvider(Protocol):
    def complete(self, prompt: str, **kwargs) -> str:
        ...

class VectorStore(Protocol):
    def add_documents(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        ...

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        ...
```

And integrations implement them:

```text
src/ragtree/integrations/llms/litellm.py
src/ragtree/integrations/vectorstores/chroma.py
src/ragtree/integrations/vectorstores/qdrant.py
src/ragtree/integrations/graphstores/neo4j.py
src/ragtree/integrations/agents/langgraph.py
```

---

## Installation

### Lightweight install

The default install should stay small:

```bash
pip install ragtree
```

For local development from source:

```bash
pip install -e ".[dev]"
```

### Optional addons

Install only the integrations you need:

```bash
pip install "ragtree[litellm]"
pip install "ragtree[chroma]"
pip install "ragtree[qdrant]"
pip install "ragtree[faiss]"
pip install "ragtree[pgvector]"
pip install "ragtree[neo4j]"
pip install "ragtree[ontology]"
pip install "ragtree[langgraph]"
pip install "ragtree[api]"
pip install "ragtree[streamlit]"
```

For a complete local stack:

```bash
pip install "ragtree[all]"
```

---

## Addon map

| Extra | Purpose | Typical modules |
|---|---|---|
| `litellm` | Multi-provider LLM and embedding calls. | `integrations/llms/litellm.py` |
| `chroma` | Local vector store demos. | `integrations/vectorstores/chroma.py` |
| `qdrant` | Vector search service integration. | `integrations/vectorstores/qdrant.py` |
| `faiss` | Local vector index. | `integrations/vectorstores/faiss.py` |
| `pgvector` | PostgreSQL vector search. | `integrations/vectorstores/pgvector.py` |
| `neo4j` | Knowledge graph export or graph store integration. | `integrations/graphstores/neo4j.py` |
| `ontology` | RDF and OWL resources. | `integrations/ontologies/` |
| `langgraph` | Agentic workflows. | `integrations/agents/langgraph.py` |
| `api` | FastAPI application surface. | `apps/api/` |
| `streamlit` | Demo and inspection UI. | `apps/streamlit/` |

---

## Testing strategy

RAGTree uses separate test layers because not all tests have the same cost.

| Test folder | Runs by default | Purpose |
|---|---:|---|
| `tests/unit/` | Yes | Pure logic: schemas, config, metrics, parsing and serialization. |
| `tests/contract/` | Yes | Protocol conformance for adapters. |
| `tests/integration/` | No | Real optional integrations such as Chroma, Qdrant, Neo4j, LiteLLM, FastAPI and Streamlit. |
| `tests/e2e/` | Selected | Tiny full pipeline tests with local fixtures. |
| `tests/regression/` | Yes | Protect current experiment outputs and relation evaluation formats. |
| `tests/fixtures/` | Data only | Tiny documents, predictions, gold labels and config files. |

Default test command:

```bash
pytest tests/unit tests/contract tests/regression
```

Integration tests:

```bash
pip install -e ".[dev,chroma,qdrant,neo4j,api,streamlit]"
pytest tests/integration
```

Docker-based integration tests:

```bash
docker compose up -d qdrant neo4j
pytest tests/integration -m docker
```

---

## Suggested pytest markers

```toml
[tool.pytest.ini_options]
markers = [
    "unit: pure unit tests with no optional dependencies",
    "contract: protocol conformance tests shared by adapters",
    "integration: tests requiring optional integrations",
    "e2e: tiny end-to-end pipeline tests",
    "regression: tests that protect experiment output formats",
    "docker: tests requiring docker compose services",
    "slow: tests that are allowed to take longer",
]
testpaths = ["tests"]
```

---

## Current experiment compatibility

The existing experiments should not be deleted during the migration. They should be preserved and gradually wrapped by the package.

| Current family | Migration target |
|---|---|
| Single-pass LLM | `RelationExtractionTask` + baseline generator. |
| ICL baseline | `RelationExtractionTask` + few-shot method. |
| CoT baseline | `RelationExtractionTask` + reasoning generator. |
| GrOWL-RAG | Ontology-guided retrieval integration. |
| KG-RAG | KG-guided retrieval integration. |
| OG-RAG | Ontology-grounded retrieval method. |
| Community KG-RAG | Graph community retrieval method. |
| Triple KG-RAG | Triple-level graph retrieval method. |
| Agentic RAG | Agent runner protocol. |
| MARAG | Multi-agent workflow integration. |
| Relation evaluation | `evaluation/relation_metrics.py` and regression tests. |
| Runtime / CO2 | Evaluation report extension. |

---

## Example target API

```python
from ragtree import RAGTreePipeline
from ragtree.tasks import RelationExtractionTask
from ragtree.integrations.llms import LiteLLMProvider
from ragtree.integrations.vectorstores import ChromaVectorStore
from ragtree.evaluation import RelationEvaluator

llm = LiteLLMProvider(model="openai/gpt-4o-mini")
vector_store = ChromaVectorStore(
    collection_name="demo_relations",
    persist_directory="./.chroma",
)

task = RelationExtractionTask(
    relation_schema=["CAUSES", "TREATS", "ASSOCIATED_WITH"],
)

pipeline = RAGTreePipeline(
    llm=llm,
    vector_store=vector_store,
    evaluator=RelationEvaluator(),
)

pipeline.index_documents([
    "Smoking is associated with lung cancer.",
    "Aspirin treats fever and pain.",
])

result = pipeline.run(task)
print(result.predictions)
print(result.evaluation)
```

---

## CI model

The CI should be split into fast default jobs and optional integration jobs.

```text
PR default:
  - install package with dev extras
  - run unit tests
  - run contract tests
  - run regression tests
  - run ruff

Optional integration jobs:
  - Chroma
  - Qdrant with Docker
  - Neo4j with Docker
  - FastAPI smoke test
  - Streamlit smoke test
  - LangGraph workflow smoke test
```

---

## Roadmap

| Sprint | Goal | Deliverables |
|---|---|---|
| 1 | Installable structure | `src/` layout, `tests/`, README, pyproject, CLI doctor. |
| 2 | Core protocols | schemas, protocols, task model, run manifest, config. |
| 3 | Adapters and apps | in-memory, LiteLLM, Chroma, FastAPI smoke, Streamlit smoke, Docker compose. |
| 4 | Experiment preservation | wrappers for current scripts, regression fixtures, CI, alpha tag. |

---

## Design rule summary

```text
core defines protocols
integrations implement protocols
tests verify protocols
examples demonstrate protocols
experiments preserve benchmark protocols
```

This is the structure that lets RAGTree grow without becoming a monolith.

---

## License

MIT.
