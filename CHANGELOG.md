# Changelog

All notable changes to RAGTree should be documented here.

## Unreleased — sprint-2/task-layer-adapters

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

## Unreleased — sprint-1/installable-core

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

## v0.1.0-alpha - Planned

- Task layer (question answering + relation extraction over the same core).
- Adapters: LLM providers, in-memory + Chroma/Qdrant vector stores, Neo4j export.
- Tiny-dataset e2e harness (causalbank, docred_causal, eventstoryline, fincausal).
- FastAPI and Streamlit surfaces, Docker profiles, experiment wrappers.
