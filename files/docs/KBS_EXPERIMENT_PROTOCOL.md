# Knowledge-Based Systems revision — executable RAGTree protocol

Paper: **KNOSYS-D-26-17743**, submitted revision benchmark. Target branch: `kbs-journal`, created from upstream `xquality`.

## Read this before spending money

The uploaded `RAGTree-xquality.zip` is an **export**, not a Git history. Its `BRANCH.txt` contains `feat/src-testable-byos-architecture`; no specific verified Git SHA is included. The original notebooks have 57 reconstructed configurations for 60 paper cells, with four missing MARAG recipes and uncertain ARAG mapping. Therefore do not claim that any inferred recipe is the exact journal run. The repository does **not** contain the full original datasets, KG indexes, ontology resources, prediction JSONLs, or local serving configuration. They must be restored on the execution machine.

The implementation adds a _separate_ KBS controller and minimal no-op hooks in the existing extraction scripts. Ordinary RAGTree invocations remain unaffected when `KBS_ACTIVE` is unset. The new controller's job identities distinguish models, phases, variants, backends, protocol mode, dataset bytes, library source hashes and resources included in fingerprints. Raw historical outputs are never overwritten by the controller.

## Install

In a clean checkout:

```bash
git fetch origin
git switch -c kbs-journal origin/xquality
# Apply the patch zip with the bundled apply_patch.py (see APPLY_PATCH.md)
python -m pip install -e .
python -m pip install -r requirements-kbs.txt
python -m pytest -q tests/kbs
```

Restore the original input JSONLs and the real KG/index/ontology resources to the paths referenced by `configs/default.yaml` and `configs/experiments/kbs_journal/recipes.json`. The program fails preflight on missing explicitly referenced resources and never fabricates them.

### LiteLLM local endpoint

Local profile uses **LiteLLM Python SDK** and OpenAI-compatible `http://localhost:8000/v1`, with the served model ID `openai/gpt-oss-20b`. LiteLLM's provider-qualified model identifier is `openai/openai/gpt-oss-20b` (the first `openai/` selects the API protocol; the remainder is the server's model ID). If your vLLM server advertises another model ID, edit `profiles.local.models` in `protocol.yaml`. `VLLM_API_KEY` is optional; otherwise `dummy` is used.

For OpenRouter set `OPENROUTER_API_KEY`, then use `--profile openrouter`. Profile model IDs are `openrouter/openai/gpt-oss-20b` and `openrouter/qwen/qwen3-30b-a3b-instruct-2507`. Verify those IDs and provider availability on your actual account before production. **Never commit API keys.**

## Three protocols (same underlying scheduler)

### Sprint 1: Smoke tests and execution integrity

```bash
python scripts/kbs_protocol.py plan
python scripts/kbs_protocol.py smoke --profile local --mode sequential --documents 1 --max-errors 3
python scripts/kbs_protocol.py status
```

By default `smoke` covers all **60 original method–dataset configurations for gpt-oss-20b**, running one deterministic sampled record in each configuration where resources and provenance are verified. Its one-record successes are stored in the same checkpoint namespace as `run` for that profile and protocol. Use `--method`, `--dataset`, `--phase`, `--model` for selected tests. A blocked or failed method is explicitly shown; it is not counted as passing. The 4 unverified MARAG recipes, ARAG mapping and inferred baseline require `--allow-unverified` after human review.

### Sprint 2: Adaptive calibration and cost estimation

```bash
python scripts/kbs_protocol.py estimate --profile local --mode sequential --samples-per-cell 6 --sample-cap 30 --max-errors 3
# After an initial local estimate, calibrate the second backbone on OpenRouter:
python scripts/kbs_protocol.py estimate --profile openrouter --phase main_qwen --samples-per-cell 6
```

The initial pilot uses length/density-stratified deterministic samples. Each selected cell expands from 6 records in steps of 6 until the one-sided bootstrap upper bound on runtime is within the configured target or the cap of 30 is reached. Actual per-call prompt/completion tokens and LiteLLM-reported cost are stored; missing usage/price is **null**, not zero. Input/output prices in `protocol.yaml` start as **null**: insert verified prices when obtaining a quote. The reported full-campaign projection covers all 192 configured jobs; if second-model output lengths are not measured, that row is labeled `secondary_model_proxy`. A **30% contingency** is applied only when a complete dollar estimate can be calculated. This is a budget planning tool, not a certified invoice.

Reports are written to `results/kbs_journal/reports/estimate_<session>.json` and `projected_campaign_<session>.json`.

### Sprint 3: Full controlled KBS benchmark

```bash
# Local sequential, one initial backbone to verify parity
python scripts/kbs_protocol.py run --profile local --phase main_gpt --mode sequential --protocol legacy_xquality

# Scientific revision on OpenRouter; runs up to five documents concurrently
python scripts/kbs_protocol.py run --profile openrouter --mode parallel --workers 5 --protocol controlled_revision_v2 --max-errors 3

# Resume pending work; by default failed records are skipped
python scripts/kbs_protocol.py resume --profile openrouter --mode parallel --workers 5 --protocol controlled_revision_v2

# Retry recorded failures explicitly
python scripts/kbs_protocol.py resume --profile openrouter --mode parallel --workers 5 --protocol controlled_revision_v2 --retry-failed

python scripts/kbs_protocol.py report --profile openrouter
```

The manuscript's expected record counts are enforced in production: MAVEN-ERE 3516, EventStoryLine 443, FinCausal 967, DocRED dev 998, CausalBank 1080. Pass `--allow-count-mismatch` for **exploration only**, never for claimed journal-complete results. **Do not use `--allow-unverified` to publish unverified historical recipes without first checking the original artifacts.**

## Frozen experimental matrix

| Phase | Method/sample specification | LLM-config jobs |
|---|---|---:|
| `main_gpt` | all 12 methods × 5 datasets, gpt-oss-20b | 60 |
| `main_qwen` | same 12 × 5, Qwen3-30B-A3B-Instruct-2507 | 60 |
| `prompt_robustness` | 4 methods × 3 datasets × 2 nonbaseline prompt variants | 24 |
| `budget_control` | GrowlRAG, CommunityRAG, SimpleAgenticRAG × 3 datasets × 2 models, max external evidence 2048 tokenizer tokens | 18 |
| `retrieval_sensitivity` | Chunk-O-RAG, CommunityRAG, Triple KG-RAG × 3 datasets × 2 alternate depth settings | 18 |
| `execution_repeats` | two agentic variants × 3 datasets × 2 extra execution IDs | 12 |
| **Total** | | **192** |

Each row is a whole method×dataset evaluation job; actual document evaluations exceed the number of jobs. Model-specific context lengths and agent LLM call counts will be measured rather than assumed. The fixed method-specific 'depth' flags are **not numerically interchangeable** (communities, chunks, and triples are different units). The final manuscript must state that distinction.

## Checkpoint and failure semantics

- One record committed in `records/<sha256(document_id)[:24]>.json` **atomically**; source document + final predictions + timing are preserved.
- SQLite `checkpoints.sqlite` uses WAL, transactions and 30 s `busy_timeout`, with status per `(run_id, doc_id)`.
- `run_id` includes the model, backend profile, phase, parameter variant, protocol mode, dataset hash, code hash and referenced resources; mixing incompatible runs is prevented.
- Successful records with intact checkpoint files are skipped on all subsequent invocations. Failed documents are skipped unless `--retry-failed` is supplied. Interrupted documents without a committed checkpoint run again.
- Transient API retries (LiteLLM: 2 default) are **not** counted as terminal record errors until exhausted.
- `--max-errors 3`: first three terminal document failures are recorded; **fourth triggers abort**. Preflight and process initialization failures are reported separately. A bounded child-process deadline defaults to 86,400 seconds per job.
- Each session keeps its own error counter; historical failure attempts remain in SQLite.
- Concurrent document processing uses separate subprocess shards (default five for cloud). Each handles its assigned documents sequentially, with no new method-internal LLM parallelism. Because some original agent strategies may use LangGraph or network tools internally, review their own concurrency behavior before increasing workers. GPU RAM/index construction is replicated across shards, so local default is **one**.
- The controller stores a shell-process log for each shard and a campaign summary, including blocked, failed, completed and partial jobs. No endless retry loops or unconditional API retry-on-resume.
- True exactly-once billing is impossible if a provider processes a request and the machine crashes before the completion response is checkpointed. The trace logs at-most local saved results, and can still overcount/undercount remote charges around process crashes.

## Original behavior versus scientific corrections

`--protocol legacy_xquality`: preserve the method inputs/selection rules, original prompt/schema logic and output parsing. The KBS LiteLLM adapter forwards documented decoding parameters, so outputs may diverge from historical vLLM behavior where those parameters were not actually forwarded. This profile is for *regression comparisons*, not proof that historical evaluation was leakage-free.

`--protocol controlled_revision_v2`: remove target `relations` (gold labels) before calling the strategy, enforce a dataset-wide relation schema, and filter provided few-shot examples whose document IDs overlap evaluation documents. The original source document remains in the checkpoint for correct gold-driven evaluation. **Important limitation:** full dataset-wide label inventories generated from the evaluation JSONL are only provisional and must be replaced with an independent official/train schema before making strong claims about leakage freedom. Some strategies construct demonstrations and external evidence in custom preparation paths; inspect these separately before declaring the full v2 benchmark independently leakage-free. In a dataset evaluated with `doc-type=all`, filtering all in-file shots may turn an ICL method effectively zero-shot. A dedicated independent training-shot pool is scientifically preferable and still needs provenance.

The `budget_control` phase trims **only the named external-evidence blocks** in three audited prompt formats. Documents, entities, demonstrations, relation schema and output instructions are untouched. The cap is equal (2,048 evidence tokens), but actual measured evidence may be lower. The token counter depends on LiteLLM's model support and must be verified on both backbones. A missing prompt boundary raises an error, rather than pretending that a full-prompt cutoff constitutes equal external retrieval. Original retrieved *candidate counts* are method-dependent; the phase is best described as evidence-*presentation*-budget matching, not equal retrieval opportunities.

The shipped observer records actual LLM messages/responses, token usage, normalization before/after where routed through the common normalizer, and final predictions; bespoke retriever candidate/triple recall and all raw tool HTTP snapshots are **not guaranteed across all 12 methods**. This limitation is explicit in `reviewer_evidence.md`. A final exhaustive failure-stage causal attribution still requires instrumenting every original retriever and parser in a way that preserves its semantics.

## Outputs and reviewer evidence

- Every job: immutable `manifest.json`, `records/`, `calls/`, `events/`, `errors/`, per-shard `.log`, and `last_execution.json`.
- Every campaign: `reports/campaign_<session>.json` + `.csv` and estimation projections.
- `report`: `metrics.csv`, `metrics.json`, `per_relation.csv`, `regression.csv`, `dataset_properties.json`, `reviewer_evidence.md`; reports identify stale source fingerprints rather than silently treating them as current.
- Scores are **gold-driven**: missing predictions are evaluated as no predicted relations (all their gold relations count FN); incomplete runs are labeled partial, not submission-ready.
- `regression.csv` references all 60 submitted paper F1 values, but the numeric difference is meaningful only after matching dataset IDs, model provider, historical resources and prompt/schema protocol.

## Validation gates before full cloud run

1. Run the shipped bounded `python -m pytest -q tests/kbs` and the local smoke protocol with real resources.
2. Resolve all unverified MARAG and ARAG source recipes; verify exact train/eval document IDs and independent few-shot pool.
3. Freeze model revision, endpoint, actual generation settings, all indexes, embedding models, source commit and allowed schema.
4. Calibrate token counting, reasoning token accounting, OpenRouter prices and cost on a paired real pilot.
5. Compare per-record predictions/contexts of the old reference and instrumented legacy profile on a fixed sample. Explain any drift.
6. Only then execute controlled v2 on both models; run uncertainty, budget and depth phases.
7. Review partial, failed and blocked cells. Do not use incomplete experiments as paper evidence.
8. Conduct and document the survey-selection protocol and adapted-method fidelity review separately; the code cannot resolve these manuscript-side reviewer requirements.

All 192 configuration **definitions** are implemented, but this is not a claim that all 192 scientific experiments were completed or that all method adapters work with absent real indexes. External GPU/OpenRouter testing is required.
