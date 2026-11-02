# KBS smoke diagnostics: FinCausal IDs + empty GPT-OSS completion

Use in the existing RAGTree `kbs` branch. Do not use `--force` or clear checkpoints.

## 1. Check why the baseline failed (read-only)

```bash
python inspect_llm_trace.py --repo .
```

The original gateway archives the full provider response **before** reporting the failure. Look at `finish_reason`, `completion_tokens`, `reasoning_tokens`, and `configured_max_tokens`. If `finish_reason=length` and output tokens are nearly all reasoning tokens, that strongly suggests exhaustion of the output allowance before the final answer. It is **not guaranteed** without inspecting the trace.

Probe the same original prompt **once** through the local vLLM endpoint (not saved as an experimental result):

```bash
python probe_saved_prompt.py --repo . --max-tokens 4096 --reasoning-effort low
```

If the probe returns a valid final answer, we should put a documented `reasoning_effort=low` and sufficient output cap **in the frozen protocol**, update fingerprints, and compare legacy versus revised scores. Do not introduce implicit retries with different settings; that would contaminate run provenance.

## 2. Correct FinCausal checkpoint IDs without deleting or merging records

First dry-run:

```bash
python rekey_fincausal.py --repo .
```

Create the derived JSONL, mapping CSV and update the one FinCausal input path in `configs/default.yaml` (original preserved and config backup saved):

```bash
python rekey_fincausal.py --repo . --apply --update-config
```

All 967 records remain present, with the original `relations`, `text`, `entities`, and `title`. For collision groups, only the **document identity** changes; the old value is stored in `kbs_original_document_id`, and a stable source line index distinguishes otherwise equal records. Noncollision identities are left unchanged. The journal's original configuration is still documented; do not apply the derived source globally without recording the difference.

Re-run smoke:

```bash
python scripts/kbs_protocol.py smoke --profile local --mode sequential \
  --method baseline --dataset fincausal --max-errors 3
```

**Note:** This script updates `configs/default.yaml` when `--update-config` is requested, so subsequent non-KBS runs will also see the derived path until you revert that one config line. Consider using a KBS-specific config override in a future orchestrator version.

## 3. Gateway trace fix

`gateway.patch` changes only the KBS LiteLLM adapter: one trace per physical API request, and a more informative error message. It **does not change model prompts, generation settings, or algorithms**.

```bash
git apply --check gateway.patch
git apply gateway.patch
python -m pytest -q tests/kbs
```

Changing code changes new KBS run fingerprints; existing checkpointed successes remain present under their earlier run ID. Existing failed checkpoints remain in SQLite and can be inspected.
