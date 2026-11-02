# RAGTree KBS — duplicate-ID checkpoint hotfix (2026-11-02)

## Why this is necessary

The controlled-revision smoke campaign passed 20/60 methods. 30 were blocked by
`document_id` collisions across EventStoryLine, DocRED, CausalBank, and ontology
annotated FinCausal, and 10 were blocked because historical commands are
unverified. The latter are **not** fixed by this patch.

## How this fix works

* The original JSONL datasets are **never modified**.
* For `controlled_revision_v2` only, duplicate-ID source files get an automatically
  generated JSONL copy in `results/kbs_journal/input_cache/`.
* Only colliding rows get a private `__kbs_record_uid` field. The original
  `document_id`, text, entities, and gold relations remain untouched. The private
  field is stripped before the document is passed into `predict_relations`.
* Ontology and KG lookups still see their original document IDs, avoiding a
  breaking change in prebuilt artifacts. Linked-data collision behavior is NOT
  scientifically corrected by this; see the caveat below.
* The SQLite checkpoints, selected IDs, sharding and evaluator use the new
  internal identity. No rows are deleted or merged.
* `legacy_xquality` continues using the original input and identifiers.
* The derived file name includes the source SHA256; a change to source bytes
  triggers a fresh cache rather than silently recycling old identities.

## Install (from project root)

```bash
cd ~/git/postdoc/ragtree
git apply --check /path/to/kbs_record_identity.patch
git apply /path/to/kbs_record_identity.patch
python -m pytest -q tests/kbs
python scripts/kbs_protocol.py plan
```

The tested checkout reported 17 passing offline tests. Your server may show
conditional skips when local datasets are present.

## Exercise a previously blocked method

```bash
python scripts/kbs_protocol.py smoke \
  --profile local --mode sequential \
  --protocol controlled_revision_v2 \
  --method baseline --dataset docred_causal --max-errors 3
```

Then run the complete baseline smoke:

```bash
python scripts/kbs_protocol.py smoke \
  --profile local --mode sequential \
  --protocol controlled_revision_v2 --max-errors 3
```

Do not use `--allow-unverified` in production until original notebook/command
provenance is established. A separate small smoke run with that flag is only a
technical execution check, not proof that a configuration matches the journal.

## IMPORTANT: scientific caveats

1. **Retrieval joins with colliding original IDs remain potentially ambiguous.**
   A method that keys ontology links or KG evidence by `document_id` may conflate
   distinct records even though its KBS checkpoint IDs are now unique. Audit
   those joins and, if needed, move to unambiguous source-title/row-level joins
   before publication. This patch fixes checkpoint integrity only.
2. The previous FinCausal script already changed the global `configs/default.yaml`
   to `fincausal_kbs_unique.jsonl`. This patch does not undo that edit. For strict
   historical comparison, consider reverting that one input path back to the
   original `fincausal.jsonl` after confirming the KBS sidecar works; that
   changes run fingerprints and may require redoing the small smoke samples.
3. Any KBS code change affects the source-code fingerprint, so existing
   completed checkpoints are preserved on disk but won't be silently reused
   under a different code version. This is intentional reproducibility safety.
4. Smoke is only one selected document per job by default. Even 60/60 smoke
   passing would not establish full-run correctness, completeness or F1 parity.

## Validation performed (no live LLM)

* 17/17 offline tests, including six identical original IDs in five parallel
  shards, atomic per-document commits and zero repeated calls on resume.
* `python -m compileall` for KBS package.
* 192 unique jobs still expand from the original protocol file.
* `git apply --check` against the prior patch state.

No actual IMMOTEP dataset or OpenRouter run was executed in this environment.
