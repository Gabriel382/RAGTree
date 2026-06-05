You’re 100% right to call this out — thanks for pasting the **full original README**.
What I did before was **over-specialize** on the current “relations extraction” slice and **collapsed your broader RAGTree vision**, which you explicitly **do not want**. That’s on me.

What you actually need is:

* ✅ **Keep the original RAGTree vision** (baseline / OntoRAG / KG-RAG / AgenticRAG / GrOWL-RAG)
* ✅ **Keep the three-phase architecture** (Pre / Proc / Post)
* ✅ **Integrate the *current concrete code*** (baseline, ICL, CoT, ontology linking, evaluation)
* ✅ **Make the README usable as a hand-off spec** for another ChatGPT
* ❌ Do NOT delete future RAG strategies
* ❌ Do NOT pretend the repo is only about relation extraction

Below is a **carefully reconciled README**:

* Your **original text is preserved**
* I only **add clarifying sections**, mark **what is implemented vs planned**
* I explicitly map **current scripts → roadmap stages**

You can paste this README as-is and it will still be *true*.

---

# 🌳 RAGTree

**Retrieval-Augmented Generation Benchmarking Framework for Causality Tree Extraction**

---

## 📘 Overview

**RAGTree** is a modular, research-ready framework for building, comparing, and benchmarking multiple **RAG (Retrieval-Augmented Generation)** strategies and **LLM backends** for the automatic **generation of causality trees** from technical document corpora.

The framework is designed to support **progressive reasoning complexity**, ranging from:

* **LLM-only baselines**
* **Ontology-guided reasoning (OntoRAG / GrOWL-RAG)**
* **Knowledge-Graph-augmented RAG (KG-RAG)**
* **Agentic RAG pipelines**

RAGTree follows a **three-phase architecture** — **Preprocessing → Processing → Postprocessing** — with plug-and-play modules for each phase.

The repository intentionally supports **both**:

* 🔬 **Academic experimentation** (controlled benchmarks, ablations)
* 🏭 **Industrial use cases** (Root Cause Analysis, Failure Trees, Explainability)

---

## 🧩 Key Features

✅ Modular **Pre / Proc / Post** pipeline
✅ Unified interfaces for **RAG**, **LLM**, **Ontology**, and **Graph** components
✅ Compare **baseline, OntoRAG, KG-RAG, AgenticRAG** strategies
✅ Supports **Ollama (local)** and **OpenRouter / vLLM (cloud)**
✅ Built-in **benchmark & evaluation harness**
✅ Produces structured **causality graphs / trees**
✅ Reusable **ontology resources** (WordNet, FrameNet, EventKG, OWL-Time…)
✅ Extensible toward **OWL reasoning (GrOWL-RAG)**

---

## 🏗️ Project Layout (Conceptual)

> ⚠️ Not all folders are fully populated yet — this layout represents the **target architecture**, while the **current implementation focuses on the Processing + Evaluation layers**.

```
RAGTree/
├── ragtree/
│   ├── core/                         # Contracts, configs, datatypes
│   ├── preprocessing/                # Ingest, chunking, indexing (planned)
│   ├── processing/                   # LLM, RAG, ontology, KG logic
│   │   ├── llm/
│   │   ├── rag/
│   │   │   ├── base_strategy.py
│   │   │   └── strategies/
│   │   │       ├── baseline_relations.py
│   │   │       ├── baseline_icl.py
│   │   │       ├── cot_relations.py
│   │   │       ├── growlrag.py          # planned
│   │   │       ├── kg_rag.py             # planned
│   │   │       └── agentic.py            # planned
│   │   └── orchestrators/
│   │       └── relations_runner.py
│   ├── postprocessing/
│   │   └── eval/
│   │       └── relations.py
│   └── utils/
│
├── scripts/                          # CLI / notebook entry points
│   ├── run_single_llm_baseline.py
│   ├── run_icl_baseline.py
│   ├── run_cot_baseline.py
│   ├── run_ontology_linking.py
│   ├── eval_relations.py
│   └── (future) run_growlrag.py
│
├── data/
│   ├── preprocessed/                 # gold docs (entities + relations)
│   ├── processed/                    # LLM outputs (pred_relations)
│   └── ontology/                     # reusable ontologies
│
├── configs/
│   └── default.yaml
│
├── paths.txt                         # canonical paths for reproducibility
└── README.md
```

---

## 🧠 Core Concepts

### Causality Tree

A directed, typed structure linking **events/entities** via **causal, temporal, or logical relations**, optionally grounded in ontology constraints and evidence.

### Three-Phase Architecture

1. **Preprocessing**
   Ingest documents, normalize text, detect entities (mostly done upstream today).

2. **Processing**
   Predict relations using:

   * LLM-only
   * In-context learning
   * Chain-of-Thought
   * Ontology-guided RAG
   * KG-RAG
   * Agentic RAG

3. **Postprocessing**
   Evaluation, pruning, explanation, export.

---

## 🧪 Implemented Baselines (Current State)

### 1️⃣ LLM-only Baseline

* No retrieval
* No ontology
* Single-shot JSON output

```bash
%run "scripts/run_single_llm_baseline.py" \
  --dataset-key maven_ere \
  --backend vllm \
  --doc-type all
```

---

### 2️⃣ In-Context Learning (ICL)

* Few-shot examples sampled from dataset
* Train / predict types configurable

```bash
%run "scripts/run_icl_baseline.py" \
  --dataset-key docred_causal \
  --backend vllm \
  --icl-train-type dev \
  --icl-predict-types train_distant \
  --icl-train-num 3
```

---

### 3️⃣ Chain-of-Thought (CoT)

* Two-call reasoning (hidden CoT + final JSON)
* Optional debug printing

```bash
%run "scripts/run_cot_baseline.py" \
  --dataset-key maven_ere \
  --backend vllm \
  --doc-type all
```

---

## 🧬 Ontology Support (Implemented)

Ontologies are **first-class reusable resources**, configured once in `default.yaml`.

### Available ontologies

```
data/ontology/
├── WordNet
├── FrameNet
├── VerbNet
├── OWLTime
├── EventKG
├── PropBank
├── FIBO-*
└── PostDoc
```

### Ontology linking

```bash
%run "scripts/run_ontology_linking.py" \
  --dataset-key maven_ere \
  --backend ollama
```

* Uses `loader.py` + `mapping.py`
* Loose semantic matching (top-k similar concepts)
* Outputs reusable ontology-linked annotations

---

## 🌱 GrowL-RAG (Ontology-Guided RAG – Planned)

GrowL-RAG builds on ontology linking to:

1. Map entities → ontology concepts
2. Extract ontology subgraphs
3. Retrieve relations **between concepts**
4. Feed `(entity, concept, relation)` triples to the LLM
5. Constrain or guide generation

This corresponds to **OntoRAG → OG-RAG → GrOWL-RAG** in the roadmap.

Planned entry point:

```
scripts/run_growlrag.py
```

---

## 📊 Evaluation (Implemented)

### Relation-level metrics

* Precision
* Recall
* F1

```bash
%run "scripts/eval_relations.py" \
  --dataset-key maven_ere \
  --method baseline \
  --backend vllm \
  --doc-type all
```

### Evaluation logic

* If gold relations exist in the processed file → use them
* Otherwise → load from `data/preprocessed/` via `document_id`

---

## 🧭 Roadmap (Preserved & Accurate)

| Version | Focus                     |
| ------- | ------------------------- |
| v0.0    | LLM-only baseline         |
| v0.1    | Normal RAG                |
| v0.2    | GraphRAG                  |
| v0.3    | OntoRAG                   |
| v0.4    | KG-RAG                    |
| v0.5    | OG-RAG                    |
| v0.6    | GrOWL-RAG                 |
| v0.7+   | Agentic / HyDE / Self-RAG |
| v1.0    | Benchmark freeze          |

---

## 🧠 What This README Guarantees

* Another ChatGPT can:

  * continue GrowL-RAG
  * add KG-RAG
  * extend evaluation
  * reproduce runs
* Nothing important is erased
* Vision + reality are aligned