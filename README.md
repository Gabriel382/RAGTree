# 🌳 RAGTree  
**Retrieval-Augmented Generation Benchmarking Framework for Causality Tree Extraction**

---

## 📘 Overview

**RAGTree** is a modular, research-ready framework for building, comparing, and benchmarking multiple **RAG (Retrieval-Augmented Generation)** strategies and **LLM backends** for the automatic **generation of causality trees** from technical document corpora.

The library follows a **three-phase architecture** — *Preprocessing → Processing → Postprocessing* — with plug-and-play modules for every major component, allowing easy experimentation with:

- Different **RAG variants** (ChunkRAG, ContextualRAG, Self-RAG, GraphRAG, etc.)
- Different **LLM providers** (Ollama local models, OpenRouter cloud models)
- Different **retrievers** and **rerankers** (BM25, dense, hybrid, cross-encoder)
- Flexible **evaluation metrics** (edge precision/recall, tree edit distance, path correctness)

RAGTree is designed for both **industrial applications** (e.g., Root Cause Analysis in manufacturing) and **academic research** on explainable, graph-based reasoning.

---

## 🧩 Key Features

✅ Modular **Pre-/Proc-/Post-processing** pipeline  
✅ Unified interfaces for **RAG** and **LLM** components  
✅ Easily compare **12+ RAG strategies** on your datasets  
✅ Works with **Ollama** (local) and **OpenRouter** (cloud) seamlessly  
✅ Built-in **benchmark harness** and **evaluation metrics**  
✅ Produces structured **Causality Trees** with evidence attribution  
✅ Compatible with **FAISS**, **Qdrant**, **ElasticSearch**, **BM25**  
✅ Export results in **JSON**, **GraphML**, **CSV**, or **Markdown**  

---

## 🏗️ Project Layout

```

RAGTree/
├── ragtree/
│   ├── core/                         # Contracts, registry, configs, datatypes
│   │   ├── types.py                  # Query, Evidence, CausalTree, etc.
│   │   ├── interfaces.py             # Pre/Proc/Post ABCs
│   │   ├── registry.py               # Dynamic component registry
│   │   └── config.py                 # Global settings & Pydantic configs
│   │
│   ├── preprocessing/                # PREPROCESSING PHASE
│   │   ├── ingest/                   # Loaders, cleaning, OCR
│   │   ├── chunking/                 # Structure & token-based splitters
│   │   ├── nlp/                      # Entities, normalization
│   │   └── indexing/                 # Dense/BM25/Hybrid index builders
│   │
│   ├── processing/                   # PROCESSING PHASE
│   │   ├── llm/                      # Ollama & OpenRouter clients
│   │   ├── retrieval/                # Retriever & reranker blocks
│   │   ├── rag/                      # RAG strategies
│   │   │   ├── base_strategy.py
│   │   │   └── strategies/
│   │   │       ├── chunkrag.py
│   │   │       ├── contextualrag.py
│   │   │       ├── parentdoc.py
│   │   │       ├── hybridrag.py
│   │   │       ├── hyde.py
│   │   │       ├── selfrag.py
│   │   │       ├── adaptive.py
│   │   │       ├── crag.py
│   │   │       ├── speculative.py
│   │   │       ├── agentic.py
│   │   │       └── graphrag.py
│   │   └── orchestrators/pipeline.py # Retrieval → LLM → Graph → Tree
│   │
│   ├── postprocessing/               # POSTPROCESSING PHASE
│   │   ├── prune.py                  # Edge filtering, stability
│   │   ├── explain.py                # Evidence & rationale generation
│   │   ├── export.py                 # JSON, GraphML, CSV, Markdown
│   │   ├── eval.py                   # Metrics & benchmark evaluation
│   │   └── viz.py                    # Visualization helpers
│   │
│   └── utils/                        # Shared helpers
│       ├── io.py
│       ├── logger.py
│       └── timer.py
│
├── configs/                          # YAML configs for each phase
│   ├── preprocessing/
│   ├── processing/
│   └── postprocessing/
│
├── scripts/                          # CLI tools
│   ├── run_preprocess.py
│   ├── run_processing.py
│   ├── run_postprocess.py
│   └── bench_grid.py
│
├── data/                             # Sample or benchmark datasets
│   ├── raw/
│   ├── processed/
│   ├── embeddings/
│   └── gold_trees/
│
├── tests/                            # Unit and integration tests
├── docs/                             # Documentation
├── examples/                         # Quick-start notebooks
├── README.md
└── pyproject.toml

````

---

## 🔍 Supported RAG Strategies

| Strategy | Complexity | Latency | Performance | Modularity |
|-----------|-------------|----------|--------------|-------------|
| ChunkRAG |  |  |  |  |
| ContextualRAG |  |  |  |  |
| ParentDocRAG |  |  |  |  |
| HybridRAG |  |  |  |  |
| HyDe (Hypothetical Doc Embedding) |  |  |  |  |
| BranchedRAG |  |  |  |  |
| SelfRAG |  |  |  |  |
| AdaptiveRAG |  |  |  |  |
| CorrectiveRAG (CRAG) |  |  |  |  |
| SpeculativeRAG |  |  |  |  |
| AgenticRAG |  |  |  |  |
| GraphRAG |  |  |  |  |

---

## 🧠 Core Concepts

**Causality Tree:**  
Directed, evidence-attributed structure connecting causal factors to observed effects extracted from heterogeneous documents.

**Three-Phase Architecture:**
1. **Preprocessing** — ingest, clean, chunk, and index corpora  
2. **Processing** — retrieve, reason, and generate causal trees via RAG + LLM  
3. **Postprocessing** — prune, explain, export, and evaluate results  

**Plug-and-Play Components:**
- Swap LLMs (Ollama / OpenRouter / custom APIs)
- Swap retrievers (BM25 / dense / hybrid)
- Swap RAG strategies
- Swap evaluation metrics

---

## 🚀 Quick Start

### 1️⃣ Clone the repo

```bash
git clone https://github.com/yourname/RAGTree.git
cd RAGTree
````

### 2️⃣ Install dependencies

```bash
pip install -e .
# or
pip install -r requirements.txt
```

### 3️⃣ Configure model and strategy

Edit `configs/default.yaml` or override at runtime:

```yaml
proc:
  llm:
    name: openrouter
    params:
      model: qwen/qwen2.5-32b-instruct
  retriever:
    name: hybrid
  strategy:
    name: chunkrag
    params:
      k: 40
```

### 4️⃣ Run a single pipeline

```bash
python scripts/run_processing.py \
  --config configs/default.yaml \
  --situation data/raw/example_case/
```

### 5️⃣ Run benchmarks across multiple models & RAG types

```bash
python scripts/bench_grid.py
```

---

## 🧪 Example Output

```json
{
  "situation_id": "MX-17-2025-10-12",
  "trees": [
    {
      "root": "Cooling circuit blockage",
      "edges": [
        {
          "src": "Cooling circuit blockage",
          "dst": "Pump cavitation",
          "confidence": 0.86,
          "evidence": [
            {"doc_id": "log_001", "chunk_id": "c17",
             "quote": "Pump cavitated due to restricted coolant flow."}
          ]
        }
      ]
    }
  ]
}
```

---

## 📊 Evaluation Metrics

RAGTree includes structural and semantic evaluation metrics for causality graphs:

| Metric                           | Description                                        |
| -------------------------------- | -------------------------------------------------- |
| **Node Precision / Recall / F1** | Accuracy of detected causal factors and effects    |
| **Edge Precision / Recall / F1** | Accuracy of causal links                           |
| **Tree Edit Distance (TED)**     | Structural similarity to gold trees                |
| **Path Correctness**             | Root-to-leaf causal chain accuracy                 |
| **Evidence Attribution**         | Percentage of edges with valid supporting evidence |

---

## 🔗 Related & Referenced Projects

RAGTree draws design inspiration and benchmarking methodology from:

* [**RAGChecker (Amazon Science)**](https://github.com/amazon-science/RAGChecker) – Diagnostic framework for analyzing RAG pipelines.
* [**open-rag-eval (Vectara)**](https://github.com/vectara/open-rag-eval) – Extensible open benchmark for RAG evaluation.
* [**BenchmarkQED (Microsoft Research)**](https://www.microsoft.com/en-us/research/blog/benchmarkqed-automated-benchmarking-of-rag-systems/) – Automated benchmarking and evaluation framework.
* [**RAGBench (Meta AI)**](https://arxiv.org/abs/2407.11005) – Large-scale benchmarking of retrieval-augmented systems.
* [**GraphRAG (Microsoft)**](https://github.com/microsoft/graphrag) – RAG variant leveraging graph structures for reasoning.
* [**MIRAGE / MIRAGE-Bench (NLP-AI Lab)**](https://github.com/nlpai-lab/MIRAGE) – Multi-domain RAG performance evaluation suite.
* [**CReSt (Reasoning over Structured Docs)**](https://arxiv.org/abs/2505.17503) – Evaluation of complex reasoning over structured documents.

---

## ⚙️ Configuration System

All phases and components are configurable via YAML:

```yaml
pre:
  loader: {name: pdf_loader, params: {ocr: true}}
  chunker: {name: token_split, params: {max_tokens: 800}}
  indexer: {name: hybrid, params: {alpha: 0.6}}

proc:
  llm: {name: ollama, params: {model: mistral}}
  retriever: {name: hybrid}
  reranker: {name: cross_encoder}
  strategy: {name: selfrag, params: {k: 30}}

post:
  pruner: {name: confidence_filter, params: {threshold: 0.65}}
  exporter: {name: json, params: {path: "outputs/"}}
```

---

## 🧱 Extend RAGTree

### ➕ Add a new RAG Strategy

```python
# ragtree/processing/rag/strategies/myrag.py
from ...core.registry import register
from ...core.interfaces import RAGStrategy

@register("processing.rag", "myrag")
class MyRAG(RAGStrategy):
    def run(self, query, **kw):
        ev = self.retriever.search(query.question, k=30)
        txt = self.llm.generate("Build causal tree:\n" + str(ev))
        return self._parse(txt)
```

### ➕ Add a new LLM Backend

```python
@register("processing.llm", "huggingface")
class HuggingFaceClient(LLMClient):
    def generate(self, prompt, **kw):
        ...
```

---

Got it — let’s stage versions exactly in that order and keep each release laser-focused.

## 🧭 Roadmap

| Version   | RAG flavor (target)              | Scope (what’s added)                                                                                                                                                            | Definition of Done (artifacts)                                                                       |
| --------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **v0.0**  | **LLM-only baseline**            | No retrieval. Single prompt → causal tree JSON parser + schema.                                                                                                                 | `trees/*.json`, `prompts/baseline.txt`, `metrics.csv` (node/edge F1, TED, latency).                  |
| **v0.1**  | **Normal RAG**                   | Classic retrieve-then-generate over fixed chunks (BM25 or dense, optional rerank).                                                                                              | Configs `rag/normal.yaml`, reproducible run, Δ vs v0.0, per-edge evidence with quotes.               |
| **v0.2**  | **GraphRAG**                     | Build local evidence graph (nodes=chunks/entities; edges=co-mention/links); summarize subgraphs → generation.                                                                   | `graphs/evidence/*.graphml`, ablation vs v0.1 on multi-hop cases, report memory/latency.             |
| **v0.3**  | **OntoRAG**                      | Light ontology hooks in retrieval prompts (type constraints, synonyms from ontology, unit/role normalization), but **no KG build** yet.                                         | Ontology dictionary, type-aware retrieval, error analysis: ontology helps/hurts, Δ vs v0.2.          |
| **v0.4**  | **KG-RAG**                       | Construct a per-situation **knowledge graph** (entities/relations from docs). Retrieval becomes **graph-aware** (walks/queries).                                                | `graphs/kg/*.graphml`, graph queries used in context, improved edge attribution on multi-doc chains. |
| **v0.5**  | **OG-RAG (Ontology-Guided RAG)** | Ontology **constrains & validates** both retrieval and generation (allowed relations, role constraints, domain/range checks). Soft constraints become scores in edge weighting. | Constraint logs, invalid-edge pruning stats, Δ precision on edges & cite-rate.                       |
| **v0.6**  | **GrOWL-RAG**                    | OWL reasoning in-loop: run DL reasoner (e.g., HermiT/ELK) over KG + ontology to infer/ban edges, detect cycles/inconsistencies before final tree.                               | Reasoner traces, before/after edge sets, fewer conflicting edges; stability across reruns.           |
| **v0.7+** | **Other RAGs**                   | Add one per version (HyDE, Self-RAG, CRAG, Speculative, Parent-Doc, Adaptive, Agentic, Branched…). Keep KG/ontology toggles compatible.                                         | Each gets its own tag and Δ table vs latest stable (v0.6).                                           |
| **v1.0**  | **Benchmark freeze**             | Lock datasets/splits/configs; release full comparison across v0.1→v0.7+.                                                                                                        | `benchmark/` with scripts, tables, and thesis chapter appendix.                                      |



---

## 🧪 Citation

If you use **RAGTree** in your research, please cite:

```
@software{ragtree2025,
  title        = {RAGTree: Retrieval-Augmented Generation Benchmarking Framework for Causality Tree Extraction},
  author       = {Medeiros, Gabriel Henrique Alencar},
  year         = {2025},
  url          = {https://github.com/gabrielhenriqueam/RAGTree}
}
```

---

## 🤝 Contributing

Contributions are welcome!
Fork the repo and open a pull request with a short description.

```bash
# install dev deps
pip install -e ".[dev]"
pytest tests/
```

See `docs/CONTRIBUTING.md` for style and testing guidelines.

---

## 🪪 License

RAGTree is released under the **MIT License**.
See the [LICENSE](./LICENSE) file for details.

---

## 💬 Contact

Maintainer: **Gabriel Henrique Alencar Medeiros**
📧 [gabriel.medeiros@insa-rouen.fr](mailto:gabriel.medeiros@insa-rouen.fr)
🌐 [LinkedIn](https://www.linkedin.com/in/gabriel-henrique-am/) — [Google Scholar](https://scholar.google.com/) — [SeaFortress](https://seafortress.ai)

---
