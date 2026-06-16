# RAGTree Architecture Design

RAGTree is a bring-your-own-stack framework for Semantic RAG over complex documents. The project should be organized around stable protocols and optional adapters, so the same core can run with different LLM providers, retrievers, vector stores, graph stores, APIs, UIs, and deployment choices.

## Core principle

The core package must remain independent from heavy integrations. Integrations are installed as optional extras and loaded lazily.

## Layer map

1. Application layer: CLI, Python API, notebooks, FastAPI, Streamlit, Docker.
2. Core kernel: schemas, protocols, config, registry, run metadata, errors.
3. Task layer: QA, relation extraction, claim verification, summarization, ontology linking, graph construction, evidence selection.
4. Pipeline layer: ingest, chunk, index, retrieve, rerank, reason, generate, validate, evaluate, export.
5. Adapter layer: LLMs, embeddings, vector stores, graph stores, ontology loaders, evaluators, exporters.
6. External stack: OpenAI-compatible APIs, Ollama, LiteLLM, Chroma, Qdrant, FAISS, Neo4j, RDFLib, FastAPI, Streamlit, Docker.

## Protocol list

- ConfigProtocol
- DocumentProtocol
- ChunkingProtocol
- EmbeddingProtocol
- VectorStoreProtocol
- GraphStoreProtocol
- OntologyProtocol
- RetrieverProtocol
- RerankerProtocol
- GeneratorProtocol
- AgentProtocol
- TaskProtocol
- EvaluatorProtocol
- ExporterProtocol
- ApplicationProtocol

## Packaging rule

Base install should be lightweight:

```bash
pip install ragtree
```

Integrations should be explicit:

```bash
pip install "ragtree[api,ui,neo4j,vector-qdrant,llm-litellm]"
```

## Migration rule

Existing scripts should not be deleted. They should be wrapped progressively by stable CLI/API commands.
