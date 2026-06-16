# Configuration

RAGTree should support both Python configuration and YAML configuration.

## Minimal true-RAG config

```yaml
project:
  name: semantic_rag_demo
  run_id: demo_001

task:
  name: question_answering
  require_evidence: true

llm:
  provider: mock
  model: local-mock
  temperature: 0.0

retriever:
  provider: in_memory
  top_k: 3

outputs:
  directory: outputs/demo
```

## Relation benchmark config

```yaml
project:
  name: relation_extraction_benchmark
  run_id: maven_ere_single_llm_001

task:
  name: relation_extraction
  relation_schema: auto

dataset:
  key: maven_ere
  doc_type: all

method:
  name: single_llm

llm:
  backend: vllm
  temperature: 0.0

outputs:
  predictions: data/processed/maven_ere/single_llm/predictions.jsonl
  evaluation: data/results/maven_ere/single_llm/evaluation.json
```

## Configuration rules

- Every run should be reproducible from a config file.
- CLI flags may override config values.
- Every run should save the resolved config.
- Duplicate YAML keys should be avoided and ideally rejected.
- Secrets must come from environment variables, not committed config files.
