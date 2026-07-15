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

The project is built around a simple rule:

> The core defines contracts. Integrations implement contracts. Tests verify contracts.

RAGTree started as a research workbench for relation extraction, ontology-guided retrieval, KG-RAG, agentic RAG, evaluation, runtime analysis and CO2 estimation. It is now an installable library with a `src/` layout, a dependency-light core of stable schemas and protocols, layered tests, and optional addons — while all research experiments remain runnable.

Users bring their own:

- LLM provider;
- embedding model;
- vector database;
- retriever;
- ontology or knowledge graph;
- evaluator;
- exporter;
- API or UI layer.

## What works today

```bash
pip install -e .                     # lightweight core + task layer + in-memory stack
ragtree doctor                       # see which optional extras are present
ragtree demo semantic-rag            # deterministic QA demo, zero extras
ragtree demo relation-extraction     # RE demo with micro P/R/F1 against gold
ragtree run --config examples/configs/semantic_rag_demo.yaml
pytest tests/unit tests/contract tests/e2e tests/regression
```

A full BYOS pipeline runs with zero optional extras:

```python
from ragtree import RAGTreePipeline
from ragtree.core.schemas import Chunk
from ragtree.integrations.llms import MockLLMProvider      # swap: LiteLLMProvider, OllamaProvider, ...
from ragtree.integrations.vectorstores import InMemoryVectorStore  # swap: Qdrant, Chroma
from ragtree.retrieval import DenseRetriever               # or Hybrid / OntologyGuided / KGGuided
from ragtree.tasks import QuestionAnsweringTask            # or RelationExtraction / Summarization / ClaimVerification

store = InMemoryVectorStore()
store.add_chunks([Chunk(id="c1", document_id="d1", text="The pump failed because the seal wore out.")])

pipeline = RAGTreePipeline(retriever=DenseRetriever(store), generator=MockLLMProvider())
result = pipeline.run(QuestionAnsweringTask("Why did the pump fail?"))
print(result.output, result.evidence)
```

Swap any component by installing an extra and changing one constructor — the
pipeline code stays identical. Any object with the right methods satisfies a
protocol (no inheritance needed), and every adapter is validated against the
shared contract suite in `tests/contract/bases.py`.

## Repository layout

```text
RAGTree/
├── src/ragtree/
│   ├── core/                  # schemas, protocols, pipeline, config, registry, errors  ← stable
│   ├── tasks/                 # QA, relation extraction, summarization, claim verification
│   ├── retrieval/             # dense, hybrid, ontology-guided, KG-guided retrievers
│   ├── integrations/          # llms, embedders, vectorstores, graphstores, ontologies, exporters
│   ├── generation/            # robust JSON extraction and normalization
│   ├── apps/                  # config runner, FastAPI service, Streamlit workbench
│   ├── cli/                   # doctor, addons, version, demo, run, evaluate, export, serve, workbench
│   ├── datasets/              # dataset loaders (research layer)
│   ├── evaluation/            # relation metrics (research layer)
│   ├── kg/                    # KG stores and retrievers (research layer)
│   ├── ontologies/            # ontology loading, linking, retrieval (research layer)
│   ├── preprocessing/         # dataset converters (research layer)
│   ├── processing/            # RAG strategies and orchestrators (research layer)
│   ├── services/              # LLM clients: ollama, openrouter, vllm, mock
│   └── vendor/                # vendored third-party graph code
├── tests/
│   ├── unit/                  # pure core logic (no extras)
│   ├── contract/              # protocol conformance bases + fakes
│   ├── integration/           # real optional stacks (arrives with adapters)
│   ├── e2e/                   # tiny full pipelines (arrives with the task layer)
│   ├── regression/            # protects experiment output formats
│   └── fixtures/              # tiny committed datasets
├── experiments/               # research notebooks and benchmark data
├── scripts/                   # current benchmark entry points (preserved)
├── examples/                  # example configs
├── configs/                   # research configuration
└── docs/                      # architecture, design, sprint plan
```

The heavyweight index-based research retrievers (chunk-ORAG, community-KG,
triple-KG) remain available to the benchmark scripts under `processing/`,
`ontologies/` and `kg/`; their protocol-level counterparts live in `retrieval/`.

## CLI, API service and workbench

The full command set (design doc, section 8.1):

```bash
ragtree doctor | addons | version
ragtree demo semantic-rag | relation-extraction
ragtree run --config examples/configs/semantic_rag_demo.yaml
ragtree evaluate --gold gold.jsonl --pred predictions.jsonl [--ignore-label null]
ragtree export --input result.json --format jsonl|csv|graph-csv --output out.csv
ragtree serve --host 0.0.0.0 --port 8000     # extra: api
ragtree workbench                            # extra: ui (Streamlit)
```

`ragtree run` executes declarative YAML configs (documents + task + llm +
retriever), writes `results.jsonl`, `metrics.json` and a reproducibility
`manifest.json`. `ragtree evaluate` runs the historical benchmark metrics
over legacy `pred_relations` outputs — old result files evaluate unchanged.

The FastAPI surface (`pip install -e ".[api]"`, then `ragtree serve`):

```text
GET  /health   GET  /version
POST /retrieve            # rank evidence over posted documents
POST /runs                # execute a task spec; returns run_id + RAGResult
GET  /runs/{run_id}
POST /evaluate            # relation metrics for predictions vs reference
```

Docker profiles (design doc, section 12):

```bash
docker compose --profile api up        # minimal API
docker compose --profile full up       # API + Qdrant + Neo4j showcase
docker compose --profile neo4j up -d && NEO4J_URI=bolt://localhost:7687 pytest -m neo4j
```

## Core protocol rule

Core modules never import optional integration SDKs. This is enforced by a
unit test (`tests/unit/test_optional_import_guard.py`), not just by convention.

The core defines protocols (see `src/ragtree/core/protocols.py`):

```python
from typing import Any, Protocol

class LLMProvider(Protocol):
    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...

class VectorStore(Protocol):
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]] | None = None) -> None: ...
    def search(self, query: str, top_k: int = 5, **filters: Any) -> list[EvidenceSpan]: ...
```

Missing optional dependencies raise a helpful error instead of an import crash:

```python
from ragtree import require_extra

require_extra("chromadb", "vector-chroma")
# MissingDependencyError: This feature requires the 'vector-chroma' extra.
# Install it with: pip install 'ragtree[vector-chroma]'
```

## Installation

Lightweight install (schemas, protocols, config, registry, CLI):

```bash
pip install -e .
```

Development install:

```bash
pip install -e ".[dev]"
```

Optional addons, by category (names match `pyproject.toml`):

| Extra | Purpose |
|---|---|
| `llm-openai`, `llm-ollama`, `llm-litellm` | LLM provider clients. |
| `embeddings` | sentence-transformers embedding backend. |
| `vector-faiss`, `vector-chroma`, `vector-qdrant`, `vector-elastic` | Vector stores. |
| `graph`, `neo4j` | Graph processing and Neo4j store/export. |
| `rdf` | RDF / OWL ontology support (rdflib, owlready2). |
| `api`, `ui` | FastAPI service and Streamlit workbench surfaces. |
| `notebooks`, `docs`, `ops`, `dev` | Tooling. |
| `all` | Everything, for a full local showcase. |

```bash
pip install -e ".[llm-litellm,vector-chroma,neo4j]"
```

`ragtree addons` prints this table with live install status.

## Testing

| Test folder | Runs by default | Purpose |
|---|---:|---|
| `tests/unit/` | Yes | Pure logic: schemas, config, registry, errors, CLI, import guard. |
| `tests/contract/` | Yes | Protocol conformance; reusable bases for every future adapter. |
| `tests/integration/` | No (markers) | Real optional stacks: Qdrant (in-process), Chroma, Neo4j, FastAPI TestClient, Streamlit, LiteLLM, rdflib. |
| `tests/e2e/` | Yes | Full pipelines over tiny fixture slices of CausalBank, DocRED, EventStoryLine and FinCausal, plus a QA corpus — deterministic mock LLM, golden metrics. |
| `tests/regression/` | Yes | Golden-metric protection of the legacy `pred_relations` format, legacy-runner/new-evaluator agreement, and script import preservation. |
| `tests/fixtures/` | Data only | Tiny committed datasets (exempt from the repo `*.json`/`*.jsonl` ignore). |

```bash
pytest tests/unit tests/contract tests/e2e   # fast, no extras required
pip install -e ".[dev,vector-qdrant,rdf]"
pytest tests/integration                     # real adapters (qdrant runs in-process)
```

## Current experiment compatibility

The research workbench is preserved, not rewritten. Scripts under `scripts/`
keep working; benchmark results live on the `xquality` branch and are never
modified. Migration targets:

| Current family | Migration target |
|---|---|
| Single-pass LLM / ICL / CoT baselines | `RelationExtractionTask` methods. |
| GrOWL-RAG, OG-RAG, Chunk-O-RAG | Ontology-guided retrieval behind `Retriever`. |
| KG-RAG, Triple KG-RAG, Community KG-RAG | KG-guided retrieval behind `Retriever`. |
| Agentic hybrid, LangGraph agents, MARAG | Agent integrations. |
| Relation evaluation | `evaluation/` + regression tests. |
| Runtime / CO2 scripts | Supplementary benchmark artifacts. |

Compatibility contracts: dataset keys stay stable, `pred_relations` outputs
stay readable, evaluation accepts old outputs and new `RAGResult` exports.

## Roadmap

Three sprints, one branch each (details in `docs/sprint-plan.md`):

| Sprint | Branch | Goal | Status |
|---|---|---|---|
| 1 | `sprint-1/installable-core` | src/ layout, core schemas + protocols, errors, test skeleton, CI. | ✅ done |
| 2 | `sprint-2/task-layer-adapters` | Task layer, adapters ported from existing code, tiny-dataset e2e harness. | ✅ done |
| 3 | `sprint-3/surfaces-alpha` | CLI command set, FastAPI/Streamlit surfaces, Docker profiles, regression layer, `v0.1.0-alpha`. | ✅ done |

## Design rule summary

```text
core defines protocols
integrations implement protocols
tests verify protocols
examples demonstrate protocols
experiments preserve benchmark protocols
```

Full architecture rationale: `docs/DESIGN.md` and the BYOS design document.

## License

MIT.
