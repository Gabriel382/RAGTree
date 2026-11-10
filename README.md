# Controlled KBS OpenRouter provider pin (GPT-OSS only)

Context: three malformed reasoning-only GPT-OSS answers came from SiliconFlow,
and one provider error from Darkbloom. The same saved failed MAVEN prompt succeeded
on DeepInfra in a diagnostic single-call replay (274 output tokens, 100 reasoning,
JSON final answer, USD 0.00012158 as recorded by OpenRouter).

This script pins DeepInfra for **GPT-OSS only**, for the **controlled_revision_v2**
KBS protocol, via OpenRouter's `provider` selection request body. Qwen routing
remains untouched. It also preserves the bounded 4096 -> 8192 on-length fallback.
The strict provider choice is traceable and versioned in `protocol.yaml`.

1. Place `install_kbs_deepinfra_routing.py` at the RAGTree root on the server.
2. In venv, run `python install_kbs_deepinfra_routing.py --repo .` (dry run).
3. Run `python install_kbs_deepinfra_routing.py --repo . --apply`.
4. Run `python -m pytest -q tests/kbs test_generation_fallback.py`.
5. Check request provenance for the new test: trace must show
   `extra_body.provider.only=['deepinfra']`; look up its OpenRouter generation
   ID if needed to verify actual backend.
6. Trial **one single document** within **one GPT-OSS method/dataset**:

   `python kbs_first_parallel_campaign_tqdm.py --execute --only-phase main_gpt --max-jobs 1 --documents-per-cell 1 --parallel-jobs 1 --workers-per-job 1 --max-errors 0`

7. Only after this passes, process the **same one cell** completely:

   `python kbs_first_parallel_campaign_tqdm.py --execute --only-phase main_gpt --max-jobs 1 --parallel-jobs 1 --workers-per-job 3 --max-errors 3 --retry-failed`

**IMPORTANT:** Versioned source/config edits change `run_id`, by design. Existing
1226 successful documents remain physically preserved in the old run and can
be used for cost analysis, but cannot be silently counted as equivalent to
the new DeepInfra-pinned run. Do not delete or rekey them. This avoids mixing
provider-routing policies in a benchmark whose reproducibility matters.

This installer has only been tested against reconstructed KBS files from the
supplied patch packages. It has not been validated against the actual IMMOTEP
working tree; dry-run and tests are mandatory. This is not a guarantee that
DeepInfra succeeds on all documents, or that LiteLLM sends `extra_body` in
all installed versions: verify the first real trace and OpenRouter provider.

There is no API key in the patch. The key stays in ~/.config/ragtree/openrouter.env.
