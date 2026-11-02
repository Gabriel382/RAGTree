# Recovery of the ten previously blocked KBS configurations

This is a **source-based reconstruction**, not an assertion that the exact final journal executions were recovered. All source paths below refer to the supplied historical `RAGTree-xquality.zip`.

## Evidence and decisions

| Configuration(s) | Decision for controlled revision | Evidence | Historical certainty |
|---|---|---|---|
| ARAG (five datasets) | `agentic_hybrid`; preserve original 5 notebook commands; `max_llm_calls=1` | `scripts/run_agentic_runtime_co2_n_docs.py`, header, `METHOD_DISPLAY`, `build_method_command`; `tests/processing/agentic.ipynb` cells 10, 22, 31, 40, 49 | **Method mapping and commands documented**; final journal run provenance unconfirmed |
| Baseline / EventStoryLine | `run_single_llm_baseline.py --dataset-key eventstoryline --backend vllm --doc-type all` | `tests/postprocessing/postprocessing_maven_ere.ipynb` cell 3 evaluates baseline; `scripts/run_supplementary_runtime_co2_v4.py` reconstructs command, marks it inferred | **Inferred run command** |
| MARAG / MAVEN-ERE | 3 shots from `train`; ontology EventKG; KG from train | `scripts/run_agentic_runtime_co2_n_docs.py` `DATASETS`; `scripts/run_missing_marag_runtime_co2.py` `MISSING_MARAG_SPECS` | **Author-documented inference**, not original journal run |
| MARAG / EventStoryLine | 3 shots from `full`; ontology owltime; KG artifact with limit=10 | Same two supplementary scripts | **Author-documented inference**; overlap/resource issue remains |
| MARAG / FinCausal | 3 shots from `train.csv`; ontology fibocoreplus; KG train.csv | Same two supplementary scripts | **Author-documented inference** |
| MARAG / CausalBank | 3 shots from `resulted_from`; ontology wordnetfull; KG resulted_from | Same two supplementary scripts | **Author-documented inference** |

For the four MARAG reconstructions, the supplementary source explicitly labels them **inferred**. Their `--shot-type` values used to be guessed as `all` in the first KBS patch; this patch corrects them to the explicitly documented values `train`, `full`, `train.csv`, `resulted_from`. All other MARAG settings remain as before, including 10 maximum LLM calls, web/Wikidata, and 3 examples.

## Access policy

- `--protocol legacy_xquality`: **only 50 directly notebook-reconstructed recipes run** by default. The other ten stay blocked, protecting strict historical claims.
- `--protocol controlled_revision_v2`: all **60 recipes may run without `--allow-unverified`** because the missing ten have explicit source references. Their manifests retain `source_documented` or `documented_inference`, so results cannot be presented as recovered historical executions.
- Truly unverified/new recipe categories remain blocked in v2 unless explicitly overridden.

This patch modifies only the KBS recipe registry, provenance gate, and tests. It does **not** change algorithms, original notebooks, datasets, resources, metrics, or existing checkpoints. Changed recipes get changed run fingerprints and therefore new independent checkpoints.

## Outstanding scientific validation before production

1. Compare original server `data/processed/*.agentic_hybrid.vllm.jsonl` and `*.marag.vllm.jsonl` with submitted 60-cell results, if still retained; notebook outputs alone cannot establish exact journal production settings.
2. Validate source KG artifacts (EventStoryLine `limit=10`) and ontology joins keyed on colliding original document IDs.
3. Audit few-shot overlap and correct source-pool separation; e.g. `full`, `resulted_from`, or `dev` may be part of evaluation set. **A passing one-record smoke does not establish no leakage.**
4. Model/provider changes and `4096` token/low-reasoning settings can alter F1. A controlled v2 result need not equal legacy or submitted paper values.
5. Full-run successes and reviewer-ready metrics still require actual 5-dataset execution and complete trace/output validation.

## Install

From repository root:

```bash
unzip RAGTree_KBS_Recipe_Recovery_20261102.zip -d /tmp/kbs-recipes
# Inspect before applying:
git apply --check /tmp/kbs-recipes/kbs_recipe_recovery.patch
git apply /tmp/kbs-recipes/kbs_recipe_recovery.patch
python -m pytest -q tests/kbs
python scripts/kbs_protocol.py plan | head -30
```

Then locally smoke the ten previously blocked configurations individually or run the complete smoke:

```bash
python scripts/kbs_protocol.py smoke --profile local --mode sequential --protocol controlled_revision_v2 --max-errors 3
```

Do not use `--allow-unverified` for that command. A run may still fail for actual model, resource, or agent network errors; this patch only resolves source-configuration gatekeeping.
