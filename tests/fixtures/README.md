# Test fixtures

Tiny committed datasets, exempt from the repo-wide `*.json`/`*.jsonl` ignore.

## relations/

Slices of the preprocessed benchmark datasets (`data/preprocessed/*.jsonl`,
built by `scripts/run_convert_dataset.py`). Slicing rule: first 5 documents
with non-empty `relations` and text under 900 characters; fields kept:
`document_id, title, text, entities, relations`. Dataset keys and the
`{TYPE: [[head_id, tail_id], ...]}` relation format are unchanged
(compatibility contract, design doc section 11.1).

| File | Source dataset |
|---|---|
| `causalbank_tiny.jsonl` | CausalBank |
| `docred_causal_tiny.jsonl` | DocRED (causal subset) |
| `eventstoryline_tiny.jsonl` | EventStoryLine (includes the `null` label) |
| `fincausal_tiny.jsonl` | FinCausal |

## qa/

`tiny_documents.jsonl`: 6 hand-written maintenance-log documents for
question answering, summarization and claim verification e2e tests.

## ontology/

`tiny_ontology.ttl`: 6 OWL classes with labels, comments and altLabels for
ontology-store and ontology-guided-retrieval tests.
