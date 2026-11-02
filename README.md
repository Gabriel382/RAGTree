# KBS controlled-revision — bounded GPT-OSS-20B generation fallback

## Observed evidence
Three *exact* saved DocRED prompts that returned no final text at the 4,096-token
output cap were replayed once with 8,192. All finished normally with
structurally valid relation JSON (1,577 / 6,649 / 542 completion tokens).
This is a diagnostic result, not a claim that their official evaluation has passed.

## Apply
From your **current** KBS branch on IMMOTEP, after all earlier hotfixes:

```bash
cd ~/git/postdoc/ragtree
git status --short
git apply --check /path/to/kbs_bounded_fallback.patch
git apply /path/to/kbs_bounded_fallback.patch
python -m pytest -q tests/kbs
```

The patch changes ONLY the KBS LiteLLM gateway, KBS campaign environment,
KBS protocol YAML, and one new test file. Existing RAG strategies are untouched.

## Policy (controlled_revision_v2, gpt_oss_20b only)

1. Run the same prompt with 4,096 maximum completion tokens and low reasoning effort.
2. If **and only if** the provider returns **no final-answer text** and
   `finish_reason=length`, make exactly **one** additional API call on the
   same prompt/messages with `max_tokens=8192`.
3. If retry succeeds, return the answer; if not, fail the record normally.
4. Both completed API attempts are independently traced with usage and cost
   (if available); a `generation_fallback` event identifies the reason.
5. Legacy xquality never enables this fallback; the Qwen model does not
   automatically use a GPT-OSS-specific recovery rule.
6. API/network-level retries remain separately capped by the current LiteLLM configuration.

## Verify on IMMOTEP

```bash
python scripts/kbs_protocol.py smoke \
  --profile local --mode sequential \
  --protocol controlled_revision_v2 \
  --method baseline --dataset docred_causal --max-errors 3

python scripts/kbs_protocol.py smoke \
  --profile local --mode sequential \
  --protocol controlled_revision_v2 \
  --method langgraph_agentic_simple --dataset docred_causal --max-errors 3
```

These smoke commands select a small document sample; they do **not** target the
three failed pilot documents or establish that all original failures are resolved
in the checkpoint database. The 3/3 successful exact-prompt diagnostic replays
are stored separately and must not be imported as canonical experiment outputs.

## Critical checkpoint warning
The existing KBS engine fingerprints all Python source plus protocol.yaml. This
patch **will change run identities**. Old results remain intact but will not be
silently reused as the same scientific run. Do not rerun the entire `estimate`
command merely to clear three historical failures. Keep its token/budget report
and account for unsuccessful calls and the extra fallbacks when finalizing costs.
When launching revised production, treat it as a newly versioned protocol.

## Limits
The fallback solves only the specific no-final-text/length case. It does NOT
retry malformed relation JSON, missing entities, network outages, or prediction
validation failures, and no extended LLM experiments were run during packaging.

## Tests
Eight isolated gateway tests exercise: first-call success, first-call content
with length finish, one bounded retry, both-attempt billing traces, no retry on
stop/empty or provider failure, failure on second attempt, and legacy isolation.
Run full `tests/kbs` suite locally before real inference.

## Precisely targeted end-to-end verification (recommended)

The ZIP also includes `verify_three_failed_docred.py`. Copy it into the
RAGTree repository root. Its default invocation is read-only:

```bash
python verify_three_failed_docred.py
```

If all three original document IDs are found exactly once, execute only those
three evaluations through the real RAGTree runner and checkpoint manager:

```bash
python verify_three_failed_docred.py --execute --profile local --max-errors 3
```

It creates/reuses the **new versioned** run ID, leaves all previous estimate
artifacts intact, and explicitly checks all three selected checkpoint statuses.
Each agentic document may make multiple LLM calls; the script does not limit it
to three *API requests*.
