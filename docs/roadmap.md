# Roadmap

The goal is to evolve the current research codebase into a professional Semantic RAG library without breaking experiments.

## Phase 0 - Documentation reset

Outcome: the repository clearly communicates that RAGTree is a Semantic RAG framework.

Deliverables:

- replace README;
- add documentation folder;
- add architecture and roadmap pages;
- document current experiment scripts;
- document compatibility promise.

## Phase 1 - Package stabilization

Outcome: the project installs and imports cleanly.

Deliverables:

- fix empty or missing core types;
- align console entry points with actual modules;
- avoid eager optional imports;
- clean duplicate config keys;
- add minimal test fixtures.

## Phase 2 - Core schemas and task model

Outcome: RAGTree has a professional API independent of relation extraction.

Deliverables:

- `Document`, `Chunk`, `Evidence`, `TaskInput`, `TaskOutput`, `RunMetadata`;
- `BaseTask`;
- `RAGTreePipeline`;
- `LLMProvider`, `Retriever`, `Evaluator`, `Exporter` interfaces.

## Phase 3 - Minimal true-RAG demo

Outcome: a recruiter can run a simple grounded RAG demo quickly.

Deliverables:

- in-memory retriever;
- mock or local provider;
- `QuestionAnsweringTask`;
- `ragtree demo semantic-rag`;
- example output with answer and evidence.

## Phase 4 - Relation extraction as first benchmark task

Outcome: current experiments are wrapped professionally.

Deliverables:

- `RelationExtractionTask`;
- CLI wrappers for current methods;
- output compatibility with `pred_relations`;
- evaluation wrapper around current relation evaluator.

## Phase 5 - Test suite and CI

Outcome: RAGTree looks like production-quality open-source software.

Deliverables:

- unit tests;
- contract tests;
- CLI smoke tests;
- GitHub Actions;
- coverage badge.

## Phase 6 - Portfolio alpha release

Outcome: publish `v0.1.0-alpha`.

Deliverables:

- release notes;
- demo video or GIF;
- architecture diagram;
- clean examples;
- LinkedIn post;
- CV bullet.

## Rule

Do not refactor all research scripts at once. Wrap first, then extract reusable logic gradually.
