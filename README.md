# 🌳 RAGTree

> RAGTree treats RAG as an end-to-end semantic processing architecture, not as a single retrieval call before a prompt.

---

## Why RAGTree exists

Many RAG prototypes are difficult to evaluate, reproduce, and integrate. They often hide retrieval quality, lack structured outputs, and depend on one specific provider or experiment script.

RAGTree is designed to make RAG systems more serious and auditable by combining:

- **retrieval**, to collect relevant evidence;
- **semantic grounding**, to connect documents with ontologies and knowledge graphs;
- **generation**, to produce answers or structured outputs;
- **reasoning**, to support more complex workflows such as CoT and agentic RAG;
- **evaluation**, to measure outputs instead of only displaying them;
- **reproducibility**, to keep datasets, configs, runs, and artifacts traceable.

---

## What RAGTree is

RAGTree is not only a relation extraction repository. Relation extraction is the first flagship benchmark task, but the architecture is intended to support several Semantic RAG tasks through shared components.

| Task family | Typical input | Typical output |
|---|---|---|
| Question answering | Query + corpus | Grounded answer + citations |
| Relation extraction | Document + relation schema | Typed relations + evidence |
| Claim verification | Claim + corpus | Supported, refuted, or insufficient evidence |
| Summarization | Document or corpus | Faithful summary + source references |
| Ontology linking | Mentions + ontology resources | Ranked concept candidates |
| Graph construction | Corpus + schema | Nodes, relations, evidence, graph exports |
| Evidence selection | Query or task request | Ranked evidence passages |

---

## Core capabilities

| Area | Capability |
|---|---|
| **Semantic retrieval** | Dense, sparse, hybrid, ontology-guided, KG-guided, chunk-level, community-level, and triple-level retrieval strategies. |
| **Grounded generation** | Answers and structured outputs that preserve evidence and provenance. |
| **Ontology and KG integration** | Reusable semantic resources for concept linking, retrieval expansion, and graph-aware reasoning. |
| **Reasoning strategies** | Baseline prompting, in-context learning, chain-of-thought, ontology-guided RAG, KG-RAG, and agentic RAG. |
| **Evaluation** | Gold-based metrics, relation metrics, evidence faithfulness, runtime tracking, and CO2 estimation. |
| **Experiment compatibility** | Existing benchmark scripts remain usable while the library is progressively professionalized. |
| **Portfolio value** | Demonstrates applied GenAI engineering, RAG architecture, semantic technologies, evaluation, and research-to-product thinking. |

---

## Current implementation status

RAGTree should be read as a project with two layers.

### Implemented research workbench

The current codebase already contains runnable or partially runnable experiment families for:

- single-pass LLM baselines;
- in-context learning;
- chain-of-thought prompting;
- ontology linking;
- GrOWL-RAG;
- KG-RAG;
- OG-RAG;
- Chunk-ORAG;
- Community KG-RAG;
- Triple KG-RAG;
- Agentic RAG;
- LangGraph-based agentic variants;
- MARAG;
- relation evaluation;
- runtime and CO2 analysis.

### Professional library direction

The target library layer is being structured around:

- stable schemas and task contracts;
- provider-agnostic adapters;
- reusable retrieval and generation components;
- reproducible configuration and run artifacts;
- typed evaluation outputs;
- documented CLI and Python API entry points;
- tests for schemas, metrics, adapters, CLI commands, and regression runs.

This distinction is intentional. RAGTree keeps its research depth while moving toward a clean, production-quality software identity.

---

## Supported experiment families

| Family | Representative entry points |
|---|---|
| Single-pass LLM | `scripts/run_single_llm_baseline.py` |
| In-context learning | `scripts/run_icl_baseline.py` |
| Chain-of-thought | `scripts/run_cot_baseline.py` |
| Ontology linking | `scripts/run_ontology_linking.py` |
| GrOWL-RAG | `scripts/run_growlrag_relations.py` |
| KG-RAG | `scripts/run_kg_rag_relations.py` |
| OG-RAG | `scripts/run_ograg_relations.py` |
| Chunk-ORAG | `scripts/run_chunk_orag_relations.py` |
| Community KG-RAG | `scripts/run_community_kgrag_relations.py` |
| Triple KG-RAG | `scripts/run_triple_kg_rag_relations.py` |
| Agentic RAG | `scripts/run_agentic_hybrid_relations.py` |
| LangGraph Agentic RAG | `scripts/run_langgraph_agentic_simple_relations.py`, `scripts/run_langgraph_agentic_hybrid_relations.py` |
| MARAG | `scripts/run_marag_relations.py` |
| Evaluation | `scripts/eval_relations.py`, `scripts/eval_relations_docred.py` |
| Runtime and CO2 | `scripts/run_supplementary_runtime_co2_v4.py` and related scripts |

---

## Architecture

RAGTree follows a layered architecture.

```text
Applications
  CLI | Python API | notebooks | future API/UI
        |
        v
RAGTree Core
  schemas | configs | run metadata | registries | interfaces
        |
        v
Task Layer
  QA | extraction | verification | summarization | ontology linking | graph construction
        |
        v
Pipeline Layer
  retrieve | rerank | reason | generate | validate | evaluate | export
        |
        v
Adapters
  LLM providers | retrievers | vector stores | graph stores | ontology resources | evaluators
```

The design rule is:

> The core should depend on stable contracts, not on a specific LLM provider, vector database, graph backend, or UI.

---

## Installation

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

During the transition toward a fully packaged library, most workflows are executed through scripts.

---

## Quickstart: current benchmark workflows

Run a single-pass LLM baseline:

```bash
python scripts/run_single_llm_baseline.py \
  --dataset-key maven_ere \
  --backend vllm \
  --doc-type all
```

Run an ICL baseline:

```bash
python scripts/run_icl_baseline.py \
  --dataset-key docred_causal \
  --backend vllm \
  --icl-train-type dev \
  --icl-predict-types train_distant \
  --icl-train-num 3
```

Run a CoT baseline:

```bash
python scripts/run_cot_baseline.py \
  --dataset-key maven_ere \
  --backend vllm \
  --doc-type all
```

Run ontology linking:

```bash
python scripts/run_ontology_linking.py \
  --dataset-key maven_ere \
  --backend ollama
```

Evaluate relation predictions:

```bash
python scripts/eval_relations.py \
  --dataset-key maven_ere \
  --method single_llm \
  --backend vllm
```

---

## Target Python API

The intended professional API should eventually look like this:

```python
from ragtree import RAGTree
from ragtree.tasks import QuestionAnsweringTask
from ragtree.retrievers import HybridRetriever
from ragtree.llms import LLMProvider

app = RAGTree(
    retriever=HybridRetriever(),
    llm=LLMProvider.from_config("configs/default.yaml"),
)

result = app.run(
    task=QuestionAnsweringTask(
        question="What evidence explains the observed failure?"
    ),
    documents=documents,
)

print(result.answer)
print(result.evidence)
print(result.metrics)
```

This API is a target interface. The current repository still exposes most functionality through experiment scripts.

---

## Repository structure

```text
RAGTree/
├── configs/                 # Configuration files
├── ragtree/                 # Library modules
│   ├── core/                # Config, interfaces, registry, core types
│   ├── datasets/            # Dataset loaders and adapters
│   ├── evaluation/          # Metrics and evaluation helpers
│   ├── kg/                  # KG retrieval and graph store utilities
│   ├── ontologies/          # Ontology loading, mapping, retrieval, linking
│   ├── postprocessing/      # Export, pruning, explanation, visualization
│   ├── preprocessing/       # Ingestion, chunking, indexing
│   └── processing/          # Prompts, RAG strategies, orchestration
├── scripts/                 # Runnable experiment entry points
├── docs/assets/             # Project images and documentation assets
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Semantic resources

RAGTree is designed to use reusable semantic resources such as:

- WordNet;
- FrameNet;
- VerbNet;
- OWL-Time;
- EventKG;
- PropBank;
- FIBO;
- custom domain ontologies;
- custom knowledge graphs.

These resources are part of the project identity. RAGTree is positioned as a Semantic RAG framework, not a generic prompt orchestration wrapper.

---

## Evaluation philosophy

RAGTree treats evaluation as a first-class component.

A serious RAG pipeline should answer questions such as:

- Did the system retrieve the right evidence?
- Is the generated output supported by retrieved evidence?
- Are structured predictions valid and comparable?
- How does the method behave across datasets?
- What is the runtime cost?
- What is the estimated energy or CO2 footprint?
- Which method is more robust under the same evaluation protocol?

For relation extraction, this currently includes micro metrics, per-label analysis, predicted relation files, and benchmark reports. The same philosophy should extend to all future Semantic RAG tasks.

---

## Roadmap

| Phase | Goal |
|---|---|
| **0. Identity cleanup** | Present RAGTree as a serious Semantic RAG framework. |
| **1. Stabilization** | Fix packaging, imports, CLI entry points, and minimal runnable examples. |
| **2. Core contracts** | Define stable schemas for documents, chunks, evidence, tasks, outputs, and runs. |
| **3. Experiment wrappers** | Preserve current scripts while wrapping them in cleaner APIs and commands. |
| **4. True RAG demo** | Add a minimal QA or grounded generation demo independent of relation extraction. |
| **5. Test suite** | Add unit, smoke, regression, adapter, and CLI tests. |
| **6. Alpha release** | Publish a portfolio-ready release with docs, examples, and reproducible workflows. |

---

## Portfolio positioning

RAGTree is designed to demonstrate:

- professional Python engineering;
- advanced RAG system design;
- ontology and knowledge graph integration;
- relation extraction and document intelligence;
- evaluation methodology;
- agentic and multi-step retrieval workflows;
- research-to-product transformation.

Interview summary:

> RAGTree started as a research benchmark for relation extraction with LLMs, ontologies, KG-RAG, and agentic RAG. I am evolving it into a professional Semantic RAG framework with reusable tasks, grounded outputs, evaluation, semantic resources, and production-oriented architecture.

---

## License

  * continue GrowL-RAG
  * add KG-RAG
  * extend evaluation
  * reproduce runs
* Nothing important is erased
* Vision + reality are aligned
