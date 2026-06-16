# Task model

RAGTree should become task-oriented. This is the key change that transforms it from a relation extraction workbench into a true Semantic RAG library.

## Base task interface

```python
class BaseTask(Protocol):
    name: str

    def build_prompt(self, input: TaskInput, context: list[Evidence]) -> str:
        ...

    def parse_output(self, raw_output: str) -> TaskOutput:
        ...

    def validate_output(self, output: TaskOutput) -> TaskOutput:
        ...
```

## Generic task flow

```mermaid
flowchart LR
    Input[Task input] --> Plan[Task plan]
    Plan --> Retrieve[Retrieve evidence]
    Retrieve --> Context[Build semantic context]
    Context --> Generate[Generate or extract]
    Generate --> Parse[Parse structured output]
    Parse --> Validate[Validate]
    Validate --> Evaluate[Evaluate]
    Evaluate --> Export[Export artifacts]
```

## Task families

| Task | Current relation to the codebase | Professional target |
|---|---|---|
| Relation extraction | Fully central today | First stable benchmark task. |
| Question answering | Not the main current focus | Minimal true-RAG demo task. |
| Claim verification | Related to no-gold evaluation | Add faithfulness and support/refute/insufficient labels. |
| Summarization | Future task | Evidence-grounded summary with citations. |
| Ontology linking | Implemented as pipeline component | Promote to explicit task. |
| Graph construction | Partially supported through KG scripts | Promote to task producing nodes and edges. |
| Evidence selection | Implicit in retrievers | Promote to standalone retriever evaluation task. |

## RelationExtractionTask

Relation extraction should remain the flagship scientific task.

Inputs:

- document;
- entity mentions;
- relation schema;
- optional ontology or KG context;
- optional few-shot examples.

Outputs:

- `pred_relations` for compatibility;
- evidence for each predicted relation;
- method metadata;
- parse diagnostics.

## QuestionAnsweringTask

Question answering should be the first general RAG demo because it is easy for recruiters to understand.

Inputs:

- question;
- documents;
- optional answer style;
- optional evidence constraints.

Outputs:

- answer;
- evidence chunks;
- confidence;
- retrieval metadata.

## ClaimVerificationTask

Claim verification is useful for no-gold evaluation and industrial audit workflows.

Outputs:

- label: `supported`, `refuted`, or `insufficient_evidence`;
- evidence;
- rationale;
- confidence.

## OntologyLinkingTask

Ontology linking turns semantic resources into reusable infrastructure.

Outputs:

- mention;
- candidate concept;
- score;
- ontology source;
- explanation.
