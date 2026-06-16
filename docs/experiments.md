# Experiments

The current experiments are an asset. They should be preserved and documented as the research benchmark layer of RAGTree.

## Current experiment families

| Method | Script | Expected role |
|---|---|---|
| Single LLM | `scripts/run_single_llm_baseline.py` | Non-RAG baseline. |
| ICL | `scripts/run_icl_baseline.py` | Few-shot baseline. |
| CoT | `scripts/run_cot_baseline.py` | Reasoning baseline. |
| Ontology linking | `scripts/run_ontology_linking.py` | Prepares ontology links for semantic retrieval. |
| GrOWL-RAG | `scripts/run_growlrag_relations.py` | Ontology-guided RAG. |
| KG-RAG | `scripts/run_kg_rag_relations.py` | Knowledge graph retrieval. |
| OG-RAG | `scripts/run_ograg_relations.py` | Ontology-grounded retrieval. |
| Chunk-ORAG | `scripts/run_chunk_orag_relations.py` | Chunked ontology retrieval. |
| Community KG-RAG | `scripts/run_community_kgrag_relations.py` | Community-level graph retrieval. |
| Triple KG-RAG | `scripts/run_triple_kg_rag_relations.py` | Triple-level graph retrieval. |
| Agentic hybrid | `scripts/run_agentic_hybrid_relations.py` | Multi-step agentic reasoning. |
| LangGraph simple | `scripts/run_langgraph_agentic_simple_relations.py` | Graph-orchestrated agentic workflow. |
| LangGraph hybrid | `scripts/run_langgraph_agentic_hybrid_relations.py` | Hybrid graph-orchestrated workflow. |
| MARAG | `scripts/run_marag_relations.py` | Multi-agent RAG. |

## Dataset keys

The documentation should preserve the dataset keys already used in scripts and notebooks:

- `maven_ere`
- `eventstoryline`
- `fincausal`
- `docred_causal`
- `causalbank`
- `maintdoc` or industrial datasets when available

## Compatibility fields

Relation extraction outputs should preserve these fields:

```json
{
  "document_id": "...",
  "type": "dev",
  "relations": {},
  "pred_relations": {},
  "method": "growlrag",
  "metadata": {}
}
```

## Professional CLI mapping

The current scripts can remain as low-level launchers while the professional CLI wraps them.

```bash
ragtree experiments relation run --dataset-key maven_ere --method single_llm --backend vllm
ragtree experiments relation run --dataset-key maven_ere --method growlrag --backend vllm
ragtree experiments relation run --dataset-key docred_causal --method marag --backend vllm
ragtree evaluate relation --dataset-key maven_ere --method growlrag --backend vllm
```

## Experiment configuration

Every benchmark run should be reproducible through a YAML file.

```yaml
project:
  name: ragtree_relation_benchmark
  run_id: maven_ere_growlrag_v001

task:
  name: relation_extraction
  relation_schema: auto
  require_evidence: true

dataset:
  key: maven_ere
  split: all
  input_path: data/preprocessed/maven_ere.jsonl

method:
  name: growlrag
  retriever: ontology
  top_k: 5

llm:
  backend: vllm
  model: gpt-oss-20b
  temperature: 0.0

outputs:
  predictions: data/processed/maven_ere/growlrag/predictions.jsonl
  report: data/results/maven_ere/growlrag/evaluation.json
```
