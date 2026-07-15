# Changelog

All notable changes to RAGTree should be documented here.

## v0.1.0-alpha — 2026-07-15

First public alpha of the bring-your-own-stack architecture: installable
core with stable schemas and protocols, task layer, adapter integrations,
retrieval layer, application surfaces (CLI, FastAPI, Streamlit), Docker
profiles and a five-layer test suite — with every research experiment
preserved. Delivered over three branches, detailed below.

## sprint-3/surfaces-alpha

### Added

- CLI command set (design §8.1): `ragtree demo semantic-rag|relation-extraction`,
  `ragtree run --config <yaml>` (writes results.jsonl, metrics.json and a
  reproducibility manifest.json), `ragtree evaluate --gold --pred`
  (historical benchmark metrics over legacy `pred_relations` outputs),
  `ragtree export --format jsonl|csv|graph-csv`, `ragtree serve`,
  `ragtree workbench`.
- `ragtree.apps.runner`: declarative config/spec -> provider, task,
  retriever, full runs; shared by CLI, API and UI. Example configs rewritten
  to be executable against the committed tiny fixtures.
- FastAPI surface (design §8.2): /health, /version, /retrieve, /runs,
  /runs/{id}, /evaluate; run registry; MissingDependencyError -> 400 with
  the pip extra named.
- Streamlit workbench: interactive QA over a pasted corpus with provider
  switching (mock/litellm/ollama).
- Regression layer: golden-metric protection of the legacy
  `pred_relations` format (full and ignore-null), proof that the legacy
  runner and the new `RelationEvaluator` agree, benchmark script import
  preservation and `__main__`-guard checks (design §11).
- Docker (design §12): compose profiles `api`, `qdrant`, `neo4j`, `full`.
- CI: regression suite + CLI demos in the core job; new `optional-surfaces`
  job (api + ui extras).

### Fixed

- Dockerfile broken by the src/ move (`COPY ragtree` -> `COPY src`); image
  now installs `.[api]` and serves the API by default.

## sprint-2/task-layer-adapters

### Added

- Task layer (`ragtree.tasks`): `BaseTask`, `QuestionAnsweringTask`,
  `RelationExtractionTask` (generic prompt mode + `results_from_strategy`
  running the 12 benchmark strategies unchanged), `SummarizationTask`,
  `ClaimVerificationTask`.
- `RAGTreePipeline` (`ragtree.core.pipeline`): retrieve → generate → parse →
  evaluate → export over protocols only; exported as `ragtree.RAGTreePipeline`.
- Integrations (`ragtree.integrations`), all lazy-importing:
  llms (Mock, Ollama, OpenRouter, vLLM, LiteLLM), embedders (Hashing,
  SentenceTransformers), vectorstores (InMemory, Qdrant incl. `:memory:`,
  Chroma), graphstores (Local wrapping `kg.local_graphstore`, Neo4j),
  ontologies (Rdflib wrapping `ontologies.loader`), exporters (Json, Jsonl,
  Csv, GraphCsv).
- Retrieval layer (`ragtree.retrieval`): Dense, Hybrid (RRF),
  OntologyGuided, KGGuided.
- `RelationEvaluator` wrapping the historical relation metrics
  (`evaluation.relations.metrics`) behind the Evaluator protocol.
- `ragtree.generation.json_utils`: robust JSON extraction + relation
  normalization shared by tasks.
- Tiny-dataset harness: committed fixture slices of CausalBank,
  DocRED-causal, EventStoryLine and FinCausal plus a QA corpus and a tiny
  ontology TTL; parametrized e2e suite with gold-echo (F1=1.0) and
  empty (F1=0.0) golden metrics; integration tests for Qdrant (in-process),
  Chroma, Neo4j, LiteLLM and the rdflib ontology store.
- Runnable zero-extras demo: `examples/semantic_rag_demo.py`.
- CI: e2e suite + demo in the core job; new `integration-light` job
  (qdrant + rdf extras); strict lint extended to the new packages.

### Changed

- `tests/contract/fakes.py` now aliases the shipped in-memory adapters —
  the reference implementations graduated into `ragtree.integrations`.
- pytest markers extended: `qdrant`, `chroma`, `rdf`.

## sprint-1/installable-core

### Added

- `src/` layout: the installable package now lives in `src/ragtree/`.
- Core contracts (`ragtree.core`): `schemas.py` (Document, Chunk, EvidenceSpan,
  RAGTask, RAGResult, RelationPrediction, RunManifest, EvaluationResult),
  `protocols.py` (LLMProvider, Embedder, VectorStore, Retriever, GraphStore,
  OntologyStore, Evaluator, Exporter), `errors.py` with `require_extra()`.
- Package exports: `from ragtree import Document, RAGTask, require_extra, __version__`.
- Layered test suite: `tests/unit` (schemas, config, registry, errors, CLI,
  optional-import guard) and `tests/contract` (reusable protocol contract bases
  plus in-memory fakes); placeholders for integration, e2e and regression layers.
- `.gitattributes` line-ending normalization; extended `.gitignore`
  (with an exemption so `tests/fixtures/` data can be committed).

### Changed

- `core/config.py`: `load_config` keeps its signature and path-resolution
  behavior but no longer assumes a repo checkout; supports explicit paths,
  `RAGTREE_CONFIG`, and upward discovery of `configs/default.yaml`, with clear
  `ConfigurationError`s.
- Research artifacts moved from `tests/` to `experiments/` (notebooks and data
  unchanged); `tests/` now holds the automated suite.
- CI runs the unit + contract suites and strict lint on the core.
- README rewritten to describe the installed reality instead of the target.

### Removed

- Empty placeholder modules (generic strategy stubs, empty preprocessing/
  postprocessing/datasets/evaluation files) and broken stubs that imported
  never-implemented types (`core/types.py`, `core/interfaces.py`,
  `processing/orchestrators/pipeline.py`, `strategies/kg_rag.py`).
  The task layer arriving in sprint 2 replaces them behind the core protocols.
- Committed `__pycache__` artifacts.

### Preserved

- Current relation extraction experiments and `scripts/` entry points.
- Existing dataset keys.
- Existing `pred_relations` evaluation format (now also mirrored by
  `ragtree.core.schemas.RelationPrediction`).
- Runtime and CO2 experiment scripts.
- Benchmark results on the `xquality` branch (untouched).
