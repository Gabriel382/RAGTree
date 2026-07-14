"""Guardrail: core and CLI must never import optional SDKs at module level.

This enforces the main design rule (BYOS architecture document, section 1):
the core defines interfaces; integrations implement them behind lazy imports.
The check is static (AST-based), so it holds regardless of what happens to be
installed in the environment running the tests.
"""

import ast
from pathlib import Path

import ragtree

FORBIDDEN_TOP_LEVEL = {
    "chromadb",
    "qdrant_client",
    "neo4j",
    "fastapi",
    "uvicorn",
    "streamlit",
    "litellm",
    "ollama",
    "openai",
    "rdflib",
    "owlready2",
    "langgraph",
    "sentence_transformers",
    "faiss",
    "elasticsearch",
    "torch",
    "networkx",
    "docker",
}

GUARDED_SUBPACKAGES = ("core", "cli")


def _module_level_import_roots(tree: ast.Module) -> set[str]:
    """Collect root names of imports outside function/class bodies."""
    roots: set[str] = set()
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # lazy imports inside defs are the sanctioned pattern
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
        stack.extend(ast.iter_child_nodes(node))
    return roots


def test_core_and_cli_do_not_import_optional_sdks_at_module_level():
    package_root = Path(ragtree.__file__).parent
    offenders: dict[str, list[str]] = {}
    for subpackage in GUARDED_SUBPACKAGES:
        for py_file in sorted((package_root / subpackage).rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            bad = _module_level_import_roots(tree) & FORBIDDEN_TOP_LEVEL
            if bad:
                offenders[str(py_file)] = sorted(bad)
    assert not offenders, (
        "Optional SDKs imported at module level in guarded packages: "
        f"{offenders}. Use ragtree.core.errors.require_extra and lazy imports."
    )


def test_importing_core_succeeds_without_optional_extras():
    # Redundant with the static check, but exercises the real import path.
    import ragtree.core  # noqa: F401
    import ragtree.core.protocols  # noqa: F401
    import ragtree.core.schemas  # noqa: F401
