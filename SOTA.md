---

## 🧪 Full Validation Pipeline — Step-by-Step Enumeration

### **I. Project setup**

1. Create a clean Python env (`conda` or `venv`).
2. Install core libs: `transformers`, `datasets`, `spacy`, `networkx`, `rdflib`, `sklearn`, `rapidfuzz`, `pandas`, `tqdm`.
3. Fix random seed (`random`, `numpy`, `torch.manual_seed`).
4. Define a global `config.yaml` with paths, dataset names, retrieval k, LLM model, ontology URIs.
5. Set up reproducible logging (`logging` + timestamped log file).
6. Create folders:
    `/data_raw`, `/data_processed`, `/models`, `/kg`, `/results`, `/jsonresults`.

---

### **II. Dataset loading & normalization**

7. For each dataset (CausalBank, EventStoryLine, MAintDoc, DocRED): download or clone official repo.
8. Read JSON/CSV into `pandas.DataFrame`.
9. Normalize column names → `text`, `entities`, `relations`.
10. Ensure unique `doc_id`.
11. Lowercase text, strip spaces.
12. Remove HTML or special chars.
13. Sentence-split with spaCy (`sentencizer`).
14. Save tokenized sentences to `/data_processed/{dataset}_sents.json`.
15. Compute token offsets for every sentence.
16. Align gold entity spans (ground truth) to offsets.
17. Align gold relations (cause/effect IDs).
18. Store all as structured JSONL.

---

### **III. Ontology / KG preparation**

19. Download the matching ontology/KG:
     • ConceptNet → CausalBank
     • FrameNet → EventStoryLine
     • ISO 15926 → MAintDoc
     • UMLS/SNOMED → DocRED
20. Convert OWL or TTL to RDFLib Graph.
21. Extract all classes and properties (URIs → labels).
22. Build a dict: `label_to_uri = {lower(label): uri}`.
23. Parse domain/range axioms → `allowed_domain_range[(prop_uri)] = (set(domain), set(range))`.
24. (Opt.) Extract causal relation subproperties (`causes`, `influences`, `leadsTo`).
25. Serialize processed ontology as `ontology.pkl` for fast load.

---

### **IV. Named-Entity Recognition**

26. Load spaCy or transformer NER model (e.g., `bert-base-cased-finetuned-ner`).
27. For each doc: run NER → candidate entity spans with label.
28. Save NER predictions (`doc_id`, `span_start`, `span_end`, `text`, `label`).
29. Match NER preds to gold entities → compute Precision, Recall, F1 (@ exact & partial match).
30. Store NER F1 per dataset in `metrics_ner.csv`.

---

### **V. Entity → Ontology concept linking**

31. For each NER entity text: normalize (lemmatize, lowercase).
32. Try exact string match in `label_to_uri`.
33. If no match, apply fuzzy match (RapidFuzz ratio ≥ 0.85).
34. If still none, query LLM: “Map ‘{entity}’ to closest concept in {ontology_name} (JSON: {label, uri})”.
35. Store mapping `entity_text → (uri, confidence)`.
36. Keep `None` for unmapped entities.
37. Compute linking accuracy = mapped / total.

---

### **VI. Retrieval modules for RAG variants**

38. Index sentences with sentence-transformer (`all-MiniLM-L6-v2`).
39. Build FAISS index for vector RAG.
40. Build co-occurrence graph (nodes = entities, edges = co-mentions).
41. Store adjacency list for GraphRAG.
42. For KG-RAG / OntoRAG, build SPARQL query wrapper around RDFLib Graph.
43. Save all retrievers in `/models/retrievers/`.

---

### **VII. Relation Extraction (LLM prompting)**

44. Define prompt template (e.g.,
     “Given the context below, list all cause → effect relations in JSON format.”).
45. Implement LLM wrapper (OpenAI / HF Transformers / Ollama / Qwen).
46. For each doc: retrieve top-k contexts based on method (Flat/Graph/KG/Onto).
47. Fill prompt with retrieved context + question.
48. Call LLM with temperature = 0, max_tokens = 512.
49. Parse LLM output (JSON list of triplets).
50. Extract ( cause_text, effect_text, relation_type ).
51. Match extracted text to NER entities (using token overlap > 0.5).
52. Assign entity URIs (from linking step) to each argument.
53. Save predictions → `jsonresults/{method}_{dataset}.jsonl`.

---

### **VIII. Ontology alignment validation**

54. Load ontology Graph + `allowed_domain_range`.
55. For each predicted triplet:
     a. If any entity unmapped → `alignment = False (reason='unmapped')`.
     b. Else check if relation URI exists → if not map to `propa:hasCausalImpact`.
     c. Check domain/range:
      • domain(entity₁_uri) ∈ allowed_domain for relation?
      • range(entity₂_uri) ∈ allowed_range?
     d. Mark `alignment=True` if both pass.
56. Record alignment result and rule violations.
57. Compute:
     - Alignment accuracy = aligned / total.
     - Consistency rate = no violations / total.

---

### **IX. Provenance / faithfulness check**

58. For each triplet, get retrieved context (text).
59. Search both entities strings within context.
60. If both found in same sentence → `faithful=True`.
61. Else run similarity (≥ 0.8) on embeddings → if ≥ threshold → `faithful=True`.
62. Compute faithfulness % = faithful / total.

---

### **X. Relation-Extraction metrics**

63. For each dataset and method:
     a. Compare predicted triplets to gold triplets (by span or mapped URIs).
     b. True Positives = exact match; False Positives = pred not in gold; False Negatives = gold not pred.
     c. Compute Precision = TP / (TP+FP); Recall = TP / (TP+FN); F1 = 2 PR / (P+R).
64. Store metrics in `results/re_metrics.csv`.
65. Compute macro & micro averages across datasets.

---

### **XI. Aggregated ontology metrics**

66. Join alignment, consistency, and faithfulness per method.
67. Compute mean ± std over datasets.
68. Output table: F1 (NER), F1 (RE), Alignment %, Consistency %, Faithfulness %.

---

### **XII. Result export & validation plots**

69. Save per-dataset JSON with metrics.
70. Plot bar charts (method vs metric), correlation heatmap (F1 vs Consistency).
71. Export to `tables/results.tex` for LaTeX inclusion.
72. Zip `jsonresults/` for Zenodo deposit.

---

### **XIII. Sanity & reproducibility checks**

73. Run 5 random docs manually → inspect triplets + ontology checks.
74. Validate scripts work with different LLMs (optional).
75. Record run time per method → efficiency metric.
76. Freeze environment (`pip freeze > requirements.txt`).
77. Push code + data to GitHub / Zenodo.
78. Generate final table for KBS Section 5.

---

### ✅ Deliverables at the end

| Output                          | Description                        |
| ------------------------------- | ---------------------------------- |
| `metrics_ner.csv`               | NER P/R/F1 per dataset             |
| `results/re_metrics.csv`        | RE P/R/F1 per dataset              |
| `results/alignment_metrics.csv` | Alignment & consistency per method |
| `jsonresults/*.jsonl`           | Triplets + alignment flags         |
| `tables/results.tex`            | Ready LaTeX table                  |
| `README.md`                     | Steps + repro instructions         |

### Folder structure

ragtree/
├── pyproject.toml
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml
│   └── datasets.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── kg/
├── docs/
├── ragtree/
│   ├── __init__.py
│   ├── config.py
│   ├── logging_utils.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── io.py
│   │   ├── text.py
│   │   └── alignment.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── causalbank.py
│   │   ├── eventstoryline.py
│   │   ├── maintdoc.py
│   │   └── docred_causal.py
│   ├── ontologies/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── mapping.py
│   │   └── constraints.py
│   ├── retrievers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── vector.py
│   │   ├── graph.py
│   │   └── kg.py
│   ├── rags/
│   │   ├── __init__.py
│   │   ├── base.py         # abstract RAG
│   │   ├── flat_rag.py     # classic RAG
│   │   ├── graph_rag.py    # GraphRAG
│   │   ├── kg_rag.py       # KG-RAG
│   │   └── onto_rag.py     # Ontology-RAG
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ner.py
│   │   └── llm.py
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── preprocess.py   # steps 7–18
│   │   ├── extract.py      # steps 44–53
│   │   └── validate.py     # steps 54–68
│   └── evaluation/
│       ├── __init__.py
│       ├── ner_eval.py
│       ├── re_eval.py
│       ├── alignment_eval.py
│       └── report.py
├── scripts/
│   ├── run_preprocess.py
│   ├── run_experiments.py
│   └── run_report.py
└── tests/
    ├── test_datasets.py
    ├── test_rags.py
    └── test_eval.py


---
