# Changelog

All notable changes to RAGTree should be documented here.

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
