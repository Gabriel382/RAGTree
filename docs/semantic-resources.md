# Semantic resources

RAGTree can use ontologies and semantic resources as retrieval and reasoning assets.

The current project resources include several ontology families, such as WordNet, FrameNet, VerbNet, PropBank, OWL-Time, EventKG, FIBO, COInd4, and dataset-specific ontologies. These should be documented as optional semantic assets rather than mandatory runtime dependencies.

## Role of semantic resources

| Resource type | Role in RAGTree |
|---|---|
| Ontologies | Define concepts, classes, properties, constraints, and domain vocabulary. |
| Knowledge graphs | Provide entity and relation context. |
| Lexical resources | Help with aliases, synonyms, frames, and concept expansion. |
| Dataset ontologies | Align task labels and benchmark semantics. |

## Professional design rule

Semantic resources should be accessed through adapters.

```python
class OntologyStore(Protocol):
    def search_concepts(self, query: str, top_k: int = 10):
        ...

    def get_fragment(self, concept_id: str):
        ...
```

## Why this matters

This allows RAGTree to support both lightweight demos and serious semantic experiments. A recruiter can run the simple demo. A researcher can run ontology-guided benchmarks.
