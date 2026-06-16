# Contributing to RAGTree

RAGTree is being developed as a professional Semantic RAG framework. Contributions should preserve two goals:

1. keep existing experiments reproducible;
2. improve the reusable library layer.

## Development setup

```bash
git clone <repo-url>
cd RAGTree
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Contribution rules

- Keep the base package lightweight.
- Do not import optional providers at module import time.
- Add tests for new schemas, adapters, and metrics.
- Preserve relation extraction output compatibility.
- Document every new method in `docs/experiments.md`.
- Add configuration examples for new pipelines.

## Pull request checklist

- [ ] The package imports cleanly.
- [ ] Tests pass.
- [ ] Documentation is updated.
- [ ] New optional dependency is added as an extra.
- [ ] Existing experiment outputs remain compatible.
- [ ] New method includes a minimal runnable example.
