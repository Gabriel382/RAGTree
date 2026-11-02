# RAGTree KBS smoke-recovery hotfix

This small hotfix addresses orchestration behavior only. It does **not** change original RAGTree extraction algorithms, prompts, or dataset records.

## Changes

1. A duplicated/missing input `document_id` blocks only the affected experiment, rather than aborting the entire smoke/estimation campaign. The source JSONL is never edited.
2. Smoke-job status is calculated from the records selected for **that experiment**, not from failures anywhere else in the session. Missing and previously failed records cannot be mislabeled successful.
3. `completed_with_failures` produces a nonzero CLI exit code, while the scheduler still processes other jobs below the failure threshold.

## Installation

Run from the **root of RAGTree** (where `ragtree/` and `scripts/` exist):

```bash
git apply --check /path/to/RAGTree_KBS_Smoke_Hotfix.patch
git apply /path/to/RAGTree_KBS_Smoke_Hotfix.patch
python -m pytest -q tests/kbs
```

If `git apply --check` fails, do not force the patch; compare your installed files to the original generated patch.

## Read-only diagnosis

```bash
python /path/to/kbs_smoke_diagnose.py --repo .
```

Prints duplicate FinCausal IDs with source lines and content hashes, latest checkpoint failure tracebacks, and current baseline/MAVEN logs. Does not change datasets or checkpoints.

## After resolving model connectivity / parser error

Retry just the failed baseline document:

```bash
python scripts/kbs_protocol.py smoke --profile local --mode sequential --method baseline --dataset maven_ere --retry-failed --max-errors 3
```

Avoid `--allow-unverified` for published reproducibility until historical recipes are confirmed.

FinCausal will remain **blocked** until its duplicated source document IDs are resolved reproducibly with no dropped rows or altered gold annotations.
