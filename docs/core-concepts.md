# Core concepts

The professional version of RAGTree should use stable concepts that apply across tasks.

## Document

A document is the original input unit. It can be a PDF page, an article, a technical manual section, a dataset item, a web page, or a structured JSON object.

Core fields:

- `id`
- `text`
- `title`
- `source`
- `metadata`

## Chunk

A chunk is a retrievable unit derived from a document. Chunks must preserve provenance.

Core fields:

- `id`
- `document_id`
- `text`
- `span`
- `section`
- `metadata`

## Evidence

Evidence is any piece of retrieved context that supports an output.

Core fields:

- `chunk_id`
- `document_id`
- `text`
- `score`
- `retriever`
- `metadata`

## Task

A task describes what RAGTree must solve. It should not define the method.

Examples:

- answer this question using evidence;
- extract relations according to a schema;
- verify whether a claim is supported;
- summarize this document with citations;
- link mentions to ontology concepts;
- construct a graph from documents.

## Method

A method describes how a task is solved.

Examples:

- single-pass LLM;
- dense RAG;
- ontology-guided RAG;
- KG-RAG;
- agentic RAG;
- multi-agent RAG.

## Prediction

A prediction is the task output. It must include method metadata and, when possible, evidence.

## EvaluationResult

An evaluation result records metrics, counts, method information, dataset information, and run metadata.

## RunMetadata

Run metadata makes experiments reproducible.

Recommended fields:

- `run_id`
- `timestamp`
- `task_name`
- `method_name`
- `dataset_key`
- `model`
- `config_hash`
- `git_commit`
- `duration_seconds`
- `estimated_tokens`
- `estimated_cost`
- `estimated_co2_kg`
