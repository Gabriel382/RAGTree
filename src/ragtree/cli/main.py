"""Base command-line interface for RAGTree.

The CLI is intentionally lightweight. It must not import heavy optional
integrations at module import time. Optional stacks are checked lazily by the
`doctor` and `addons` commands.
"""

from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError, version

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="RAGTree command-line interface.")
console = Console()

OPTIONAL_MODULES: dict[str, list[str]] = {
    "llm-openai": ["openai"],
    "llm-ollama": ["ollama"],
    "llm-litellm": ["litellm"],
    "embeddings": ["sentence_transformers"],
    "vector-faiss": ["faiss"],
    "vector-chroma": ["chromadb"],
    "vector-qdrant": ["qdrant_client"],
    "vector-elastic": ["elasticsearch"],
    "graph": ["networkx"],
    "neo4j": ["neo4j"],
    "rdf": ["rdflib", "owlready2"],
    "api": ["fastapi", "uvicorn"],
    "ui": ["streamlit"],
    "notebooks": ["jupyter", "ipykernel"],
    "docs": ["mkdocs"],
    "ops": ["docker"],
    "dev": ["pytest", "ruff", "mypy"],
}


def _is_available(module_name: str) -> bool:
    """Return True when a module can be imported without importing it."""
    return importlib.util.find_spec(module_name) is not None


@app.command("version")
def show_version() -> None:
    """Show the installed RAGTree package version."""
    try:
        installed_version = version("ragtree")
    except PackageNotFoundError:
        installed_version = "editable/local"
    console.print(f"RAGTree {installed_version}")


@app.command("addons")
def show_addons() -> None:
    """List optional extras and their Python import checks."""
    table = Table(title="RAGTree optional extras")
    table.add_column("Extra")
    table.add_column("Modules")
    table.add_column("Installed")

    for extra, modules in OPTIONAL_MODULES.items():
        installed = all(_is_available(module) for module in modules)
        table.add_row(extra, ", ".join(modules), "yes" if installed else "no")

    console.print(table)


@app.command("doctor")
def doctor() -> None:
    """Check the base installation and optional integration availability."""
    console.print("[bold]RAGTree installation check[/bold]")
    show_version()
    show_addons()
    console.print("\nBase CLI is available. Current benchmark workflows remain under scripts/.")


# Register user-facing commands (demo, run, evaluate, export, serve, workbench).
from ragtree.cli import commands as _commands  # noqa: E402

for _command in (
    _commands.demo,
    _commands.run,
    _commands.evaluate,
    _commands.export,
    _commands.serve,
    _commands.workbench,
):
    app.command()(_command)

if __name__ == "__main__":
    app()
