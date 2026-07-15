"""Regression: the benchmark scripts' package imports must keep resolving.

Scripts are entry points for reproducing published results; they may need
heavy data/services to RUN, but every ``ragtree.*`` module they import must
still exist after refactoring. This catches accidental deletion/renaming of
modules the research layer depends on (design doc, section 11).
"""

import ast
import importlib.util
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parents[2] / "scripts"

# Design doc section 11: entry points whose CLI behavior must not disappear.
EXPECTED_SCRIPTS = [
    "run_single_llm_baseline.py",
    "run_icl_baseline.py",
    "run_cot_baseline.py",
    "run_growlrag_relations.py",
    "run_kg_rag_relations.py",
    "run_ograg_relations.py",
    "run_chunk_orag_relations.py",
    "run_community_kgrag_relations.py",
    "run_triple_kg_rag_relations.py",
    "run_agentic_hybrid_relations.py",
    "run_marag_relations.py",
    "eval_relations.py",
]


def _ragtree_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                alias.name for alias in node.names if alias.name.startswith("ragtree")
            )
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module and node.module.startswith("ragtree"):
                modules.add(node.module)
    return modules


def test_design_entry_points_still_exist():
    missing = [name for name in EXPECTED_SCRIPTS if not (SCRIPTS_DIR / name).is_file()]
    assert not missing, f"benchmark entry points disappeared: {missing}"


def test_all_script_ragtree_imports_resolve():
    unresolved: dict[str, list[str]] = {}
    for script in sorted(SCRIPTS_DIR.glob("*.py")):
        bad = [
            module
            for module in _ragtree_imports(script)
            if importlib.util.find_spec(module) is None
        ]
        if bad:
            unresolved[script.name] = sorted(bad)
    assert not unresolved, f"scripts import missing ragtree modules: {unresolved}"


def test_scripts_are_main_guarded_where_expected():
    # Entry points should not execute work at import time.
    for name in EXPECTED_SCRIPTS:
        source = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        if source.strip():
            assert '__main__' in source, f"{name} lost its __main__ guard"
