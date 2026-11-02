# KBS patch — validation scope

**Completed in patch build:**

- 60 historical recipe entries assembled from the notebook-reference audit; four MARAG non-DocRED entries explicitly inferred, plus provisional ARAG mapping. These are marked `unverified`, rather than advertised as historically confirmed.
- 192 unique configuration identities (60 + 60 + 24 + 18 + 18 + 12).
- All recipe CLI flags statically checked against argparse arguments in the actual xquality scripts.
- New LiteLLM client integrated with the existing RAGTree service factory; original services lazily import optional Ollama/OpenRouter/vLLM dependencies.
- Original method code paths use an optional KBS call wrapper for document-atomic checkpoints. Outside KBS mode, this code is a no-op.
- SQLite WAL status/failed state; skip completed; opt-in retry failed; deterministic disjoint shards; finite process timeout; cancel when four terminal errors occur with default maximum three.
- Stratified sampling and adaptive estimates, source/data/code fingerprinted run identity, output tokens/calls/cost traces.
- Three explicit evidence-prompt token boundaries with small deterministic unit tests.
- Gold-driven micro/macro/per-label metrics, bootstrap intervals, cost, dataset characteristic and journal regression reporting.
- Bounded offline pytest suite: **14 passed**, using mock LiteLLM and a temporary synthetic dataset, including both sequential and 5-worker process integrations and forced API failures.

**Cannot be validated from the provided ZIP:**

- Live vLLM/IMMOTEP inference, LiteLLM/OpenRouter installed SDK behavior on the real server, current provider pricing, model IDs actually offered by the account, actual provider token cost and reasoning-mode consistency.
- End-to-end correctness of all 12 methods and all five datasets because real normalized datasets, KG/ontology indexes, embeddings and original output snapshots are not supplied.
- Exact 60 original journal run commands: notebook evidence reconstructs only 57 and some are partial/inconsistent with the submitted paper.
- End-to-end independently leak-free controlled v2 experiments; method-specific few-shot sources and KG provenance require audit.
- Exact retrieval candidate-to-prompt-to-generation loss attribution for bespoke strategies. The common trace captures all client calls but method-level evidence candidate traces are incomplete.
- Authoritative OpenRouter financial budget: local pilot and prices must be supplied.

**Do not claim that all three experimental sprints have been empirically executed.** The patch implements the orchestration and report-generating functionality for them, with offline validations; real experiments remain to be run by the user on available infrastructure.
