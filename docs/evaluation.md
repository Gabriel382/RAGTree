# Evaluation

Evaluation is one of the strongest differentiators of RAGTree. The library should make RAG measurable, not only demonstrable.

## Evaluation types

| Type | Purpose | Examples |
|---|---|---|
| Gold-based evaluation | Compare predictions to annotated ground truth. | Relation F1, Hits@k, exact match. |
| No-gold evaluation | Evaluate when labels are unavailable. | LLM-as-judge, evidence faithfulness, source grounding. |
| Retrieval evaluation | Measure evidence selection quality. | Recall@k, precision@k, nDCG. |
| Output evaluation | Measure answer or extraction quality. | F1, accuracy, semantic equivalence. |
| System evaluation | Measure engineering behavior. | Runtime, token count, cost, CO2, failure rate. |
| Reproducibility evaluation | Check run consistency. | Config hash, seed, artifact validation. |

## Relation extraction metrics

The current relation evaluation layer should remain the first stable metric module.

Recommended metrics:

- micro precision;
- micro recall;
- micro F1;
- per-label precision, recall, and F1;
- counts: true positives, false positives, false negatives;
- ignored labels when needed.

## True-RAG metrics

For general Semantic RAG tasks, add the following metrics progressively:

| Metric | Task family | Meaning |
|---|---|---|
| Evidence recall@k | QA, extraction, verification | Whether the retriever found useful context. |
| Evidence precision@k | QA, extraction, verification | Whether retrieved evidence is mostly relevant. |
| Faithfulness | QA, summarization, verification | Whether output is supported by evidence. |
| Citation coverage | QA, summarization | Whether claims are cited. |
| Answer correctness | QA | Whether the answer is correct. |
| Structured validity | Extraction, graph construction | Whether output respects the schema. |
| Graph consistency | Graph construction | Whether edges and nodes satisfy constraints. |
| Judge agreement | No-gold workflows | Whether different judges agree. |

## Evaluation artifact

All evaluators should return a common artifact:

```json
{
  "task": "relation_extraction",
  "method": "growlrag",
  "dataset": "maven_ere",
  "metrics": {
    "micro_precision": 0.0,
    "micro_recall": 0.0,
    "micro_f1": 0.0
  },
  "counts": {
    "tp": 0,
    "fp": 0,
    "fn": 0
  },
  "run_metadata": {
    "run_id": "...",
    "duration_seconds": 0.0,
    "model": "..."
  }
}
```

## Portfolio message

Many GenAI demos stop at a nice answer. RAGTree should show that professional GenAI systems need evidence, metrics, reproducibility, and failure analysis.
