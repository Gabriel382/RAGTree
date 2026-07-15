# Sprint plan: BYOS migration

Operationalizes `RAGTree_Extended_BYOS_Installable_Architecture_Design.docx` (and condenses `docs/roadmap.md`) into the fewest sprints that still cover the whole design. Each sprint lands on its own branch: `sprint-1/installable-core` (**✅ done**), `sprint-2/task-layer-adapters` (**✅ done**), `sprint-3/surfaces-alpha` (**✅ done — v0.1.0-alpha**). `xquality` is never touched.

## 1. Where the code stands today

Three categories, verified file by file:

**Already done (do not redo).** `pyproject.toml` with BYOS extras (`llm-*`, `vector-*`, `neo4j`, `rdf`, `api`, `ui`, `dev`, `all`), console script, pytest markers; CLI with `doctor` / `addons` / `version` (lazy imports, matches design §17.2); BYOS docs (`DESIGN.md`, `roadmap.md`, `ARCHITECTURE_TESTING.md`); example configs (`examples/configs/semantic_rag_demo.yaml`, `relation_extraction_benchmark.yaml`); basic CI workflow. This is roughly 60% of the design's Sprint 1 — which is why 3 sprints suffice instead of the doc's 4.

**Working code to reuse, not rewrite (~8–9k LOC).**

| Existing | Role | Target home (design §5) |
|---|---|---|
| `processing/rag/strategies/*_relations.py`, `baseline_*`, `chain_of_thought` (12 real strategies) | relation-extraction methods | stay as method implementations behind `RelationExtractionTask` |
| `processing/rag/base_strategy.py` | LLM plumbing + relation normalization | internals of the RE task; keep `predict_relations` contract |
| `services/llm/` (ollama, openrouter, vllm, mock + `get_llm_client`) | LLM clients | `integrations/llms/` adapters implementing `LLMProvider` |
| `ontologies/` (loader, mapping, linking, subontology, chunk_orag_retriever) | ontology stack | `retrieval/ontology_guided.py` + `integrations/ontologies/` |
| `kg/` (local_graphstore, document_kg, community + triple retrievers) | KG stack | `retrieval/kg_guided.py` + `integrations/graphstores/local.py` |
| `evaluation/relations/{io,metrics,runner}.py` | relation metrics | `evaluation/relation_metrics.py` |
| `preprocessing/ingest/converters/` (causalbank, docred_causal, eventstoryline, fincausal, maven_ere) + `datasets/` | dataset loading | kept; feeds the tiny-fixture test harness |
| `processing/orchestrators/relations_runner.py` | benchmark orchestration | wrapped by CLI / experiment wrappers |
| `scripts/` (~30 entry points) | reproducibility | kept runnable; become thin wrappers (design §11) |
| `vendor/byokg/` | vendored graph code | untouched; mine later for Neo4j/graph adapters |

**Empty placeholders (0 bytes) — delete, don't migrate.** All generic strategies (`chunkrag`, `selfrag`, `hyde`, `crag`, `graphrag`, `hybridrag`, `adaptive`, `agentic`, `contextualrag`, `parentdoc`, `speculative`), all of `processing/llm/`, `processing/retrieval/` (bm25, dense, hybrid, rerankers), all of `preprocessing/{chunking,indexing,nlp}` + `ingest/{clean,loader}`, `postprocessing/{explain,export,prune,viz}`, `evaluation/{ner,re,alignment}_eval`, `datasets/{docred_causal,eventstoryline,maintdoc}`, `core/types.py`, `utils/{io,timer}`. "RAG for everything" comes from the new task layer, not from filling these files.

**Known breakage to fix first:** `core/types.py` is empty, so `ragtree.core.interfaces` raises ImportError (`import ragtree` and the CLI work). `tests/` holds notebooks and dataset dumps, not pytest. `.gitattributes` is empty — the working tree shows ~4.4k CRLF-only diffs that will pollute migration commits.

## 2. Genuinely new code (all thin)

`core/schemas.py` (Document, Chunk, EvidenceSpan, RAGTask, RAGResult, RelationPrediction, RunManifest, EvaluationResult — design §6.1); `core/protocols.py` (§6.2); `core/pipeline.py` (~30 lines, §17.1); `tasks/` base + task classes (mostly prompt template + output schema each); `integrations/vectorstores/memory.py`; one real vector-store adapter (Chroma or Qdrant); Neo4j export; FastAPI + Streamlit surfaces; the pytest tree. Everything else is `git mv` plus import fixes.

## 3. The three sprints

### Sprint 1 — Installable core with contracts ✅ (branch `sprint-1/installable-core`)

Goal: `pip install -e .` from clean env; core imports never fail; test skeleton green in CI.

- Commit hygiene first: populate `.gitattributes` (`* text=auto`), normalize line endings in one dedicated commit so real diffs stay readable.
- `git mv ragtree src/ragtree` (history-preserving), set `packages.find where = ["src"]`; delete the 0-byte placeholders in the same commit.
- Move research notebooks/data out of `tests/` into `experiments/`; `tests/` becomes `unit/ contract/ integration/ e2e/ regression/ fixtures/`.
- Write `core/schemas.py`, `core/protocols.py`, `core/errors.py`, `require_extra()` helper (§7.3); fix or retire `core/types.py` + `interfaces.py` (keep `registry.py` as-is — it already works).
- First tests: unit (schemas, config, registry) + shared contract-test bases for LLMProvider/VectorStore/Retriever, run against mock implementations.
- DoD (= design Sprint 1+2): clean-env install, `ragtree doctor`, `pytest tests/unit tests/contract` pass with zero extras installed; CI green.

### Sprint 2 — Task layer + adapters ported from existing code, tiny-dataset harness ✅ (branch `sprint-2/task-layer-adapters`)

> Scope note (delivered): protocol-level ontology/KG retrievers wrap OntologyStore/GraphStore; the index-based research retrievers (chunk-ORAG, community-KG, triple-KG) stay under the research layers for scripts and are wrapped in sprint 3 alongside the experiment wrappers.

Goal: the BYOS example from design §17.3 runs end to end; RAG works for more than relations.

- Adapters by porting: `services/llm/*` → `integrations/llms/` (add LiteLLM — new but small); `local_graphstore` → `integrations/graphstores/local.py`; ontology loader → `integrations/ontologies/rdflib_store.py`. New: `InMemoryVectorStore`, one of Chroma/Qdrant, Neo4j upsert/export.
- Retrieval layer: port chunk_orag, subontology, community-KG, triple-KG retrievers behind the `Retriever` protocol (`ontology_guided.py`, `kg_guided.py`); dense/hybrid built on the vector-store protocol.
- Task layer: `tasks/base.py`; `RelationExtractionTask` wrapping the 12 existing strategies unchanged (adapter around `predict_relations`); `QuestionAnsweringTask` (new, thin) proving generality. Claim verification, summarization, ontology linking (reusing `ontologies/linking/`), graph construction (reusing `kg/document_kg.py`) are each a small subclass — include as many as time allows, they are not blockers.
- `core/pipeline.py` wiring retrieve → generate → validate → evaluate → export; `RAGResult` exporters (jsonl, csv, graph-csv from postprocessing intent).
- Automatic tests on small datasets: commit tiny fixture slices (5–10 docs) produced by the existing converters — causalbank, docred_causal, eventstoryline, fincausal — plus a tiny QA corpus and `tiny_ontology.ttl`. One parametrized e2e suite runs each task on each applicable fixture with the mock LLM and asserts golden metrics. Contract tests run per adapter; integration tests gated by markers/extras.
- DoD (= design Sprint 3 core): §17.3 example runs; `pytest` passes core-only; `pytest -m integration` passes with extras; every fixture dataset exercised in CI.

### Sprint 3 — Surfaces, experiment preservation, alpha release ✅ (branch `sprint-3/surfaces-alpha`, tag `v0.1.0-alpha`)

> Scope note (delivered): scripts already delegate to package modules, so "wrapping" is enforced by regression tests (import resolution + golden metrics on legacy outputs) rather than rewriting 30 entry points; the experiment CLI passthrough can follow post-alpha without breaking anything.

Goal: design §13.1 release checklist satisfied; `v0.1.0-alpha` tagged.

- FastAPI app (`/health`, `/version`, `/runs`, `/retrieve`, `/evaluate` — §8.2), Streamlit workbench, `ragtree run / demo / evaluate / export / serve` CLI commands on top of the pipeline.
- Docker profiles (§12): compose services for Qdrant/Neo4j/API; docker-marked integration tests.
- Experiment preservation (§11): each `scripts/run_*.py` becomes a thin wrapper calling package code, CLI flags unchanged; regression tests assert old `pred_relations` outputs still evaluate identically (fixture from a saved xquality-era output — read from the branch, never modified).
- CI matrix: fast core job + optional-extras jobs (§13); docs refresh; CHANGELOG; tag.
- DoD: release checklist §13.1, all five test layers populated, scripts still run.

## 4. Guardrails

`xquality` branch is read-only reference. Dataset keys and `pred_relations` output fields stay stable (design §11.1). Core never imports optional SDKs — enforced by a unit test that imports every `src/ragtree/core|tasks|retrieval|evaluation` module