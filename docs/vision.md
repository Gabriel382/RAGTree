# Vision

RAGTree should be presented as a professional Semantic RAG framework, not as a one-off relation extraction repository.

The project has a strong research foundation: it already contains multiple relation extraction methods, dataset converters, ontology resources, KG retrieval, ontology retrieval, agentic workflows, multi-agent workflows, evaluation, and runtime/CO2 scripts. The documentation redesign should make this foundation understandable to recruiters, engineers, researchers, and future collaborators.

## Product thesis

RAGTree treats Retrieval-Augmented Generation as a full semantic pipeline:

1. documents are normalized;
2. evidence is retrieved;
3. semantic context is constructed;
4. an LLM or agentic workflow produces an output;
5. the output is validated;
6. the result is evaluated;
7. artifacts are exported for downstream use.

This framing makes RAGTree broader than relation extraction. Relation extraction remains a central benchmark task, but the library should also support question answering, claim verification, summarization, ontology linking, graph construction, and evidence selection.

## Strategic goal

RAGTree should become a flagship portfolio project that proves four competencies at once:

| Competency | How RAGTree demonstrates it |
|---|---|
| GenAI engineering | LLM providers, RAG pipelines, task abstraction, outputs, evaluation. |
| Semantic AI | Ontologies, knowledge graphs, concept linking, evidence grounding. |
| Research quality | Benchmarks, ablations, method comparison, reproducible experiments. |
| Software engineering | Modular interfaces, tests, configuration, documentation, release strategy. |

## One-line positioning

RAGTree is a professional Semantic RAG framework for building, evaluating, and comparing grounded AI pipelines over complex documents.

## What RAGTree is

- a framework for grounded document AI;
- a benchmark workbench for RAG methods;
- a semantic layer for retrieval, ontology and KG reasoning;
- a reusable software library with stable contracts;
- a portfolio project that shows applied AI engineering maturity.

## What RAGTree is not

- not only a relation extraction script collection;
- not only a notebook repository;
- not only a paper reproduction repository;
- not only a vector search wrapper;
- not a monolithic application tied to one provider.

## Portfolio promise

A reviewer should understand RAGTree in less than 30 seconds, run a minimal demo in less than 10 minutes, and see the research benchmark layer within the first repository navigation.
