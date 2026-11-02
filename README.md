# KBS GPT-OSS-20B generation fix (2026-11-02)

This is a **controlled-revision-only** generation settings change. The original
`configs/default.yaml` and algorithms are unchanged. For the revised scientific
benchmark we configure `max_tokens: 4096` and `reasoning_effort: low` for
GPT-OSS-20B, and `max_tokens: 4096` for the second backbone, in
`configs/experiments/kbs_journal/protocol.yaml`. The controlled-run overlay
uses these settings, and the LiteLLM gateway forwards `reasoning_effort`.

## Apply
From the RAGTree Git repository, after applying the earlier gateway diagnostic
patch (the one with improved failure tracing):

```bash
git apply --check /path/to/kbs_generation_hotfix.patch
git apply /path/to/kbs_generation_hotfix.patch
python -m pytest -q tests/kbs
python scripts/kbs_protocol.py smoke --profile local --mode sequential \
  --method baseline --dataset maven_ere \
  --protocol controlled_revision_v2 --max-errors 3
```

### Warnings
- **Do not** use `--allow-unverified` by default.
- Results under `controlled_revision_v2` are *new* and not guaranteed to
  reproduce the original historical journal numerical values.
- The original legacy/controlled protocols have separate identities, and the
  changed source/manifest hashes produce new checkpoint identities.
- Test the exact LiteLLM stack with real vLLM. The earlier successful
  diagnostic called vLLM directly, not LiteLLM.
- FinCausal ID rekeying is a separate already completed operation.
- Historical configuration gaps remain blocked until reviewed.
