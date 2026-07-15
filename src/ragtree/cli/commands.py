# ragtree/cli/commands.py
"""User-facing CLI commands (design doc, section 8.1).

Registered on the main Typer app. Module-level imports stay dependency-light;
optional surfaces (FastAPI, Streamlit) are checked with require_extra inside
the commands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ragtree.core.errors import RagtreeError, require_extra
from ragtree.core.pipeline import RAGTreePipeline
from ragtree.core.schemas import Chunk, RAGResult

console = Console()

# ----------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------

_DEMO_CORPUS = [
    ("maint-001", "Pump P-102 failed on 12 March because the mechanical seal wore out."),
    ("maint-002", "Routine maintenance was performed in June on all centrifugal pumps."),
    ("maint-003", "Alarm 7741 was triggered by a pressure spike in line L-3."),
]

_DEMO_RE_DOC = {
    "document_id": "demo-doc-1",
    "title": "Pump failure report",
    "text": (
        "The pump failed because the seal wore out. "
        "The alarm was triggered by a pressure spike."
    ),
    "entities": {
        "EVENT_seal_wear": {"type": "EVENT", "mentions": [{"trigger_word": "seal wore out"}]},
        "EVENT_pump_failure": {"type": "EVENT", "mentions": [{"trigger_word": "pump failed"}]},
        "EVENT_pressure_spike": {"type": "EVENT", "mentions": [{"trigger_word": "pressure spike"}]},
        "EVENT_alarm": {"type": "EVENT", "mentions": [{"trigger_word": "alarm"}]},
    },
    "relations": {
        "CAUSES": [
            ["EVENT_seal_wear", "EVENT_pump_failure"],
            ["EVENT_pressure_spike", "EVENT_alarm"],
        ]
    },
}


def _demo_semantic_rag() -> None:
    from ragtree.integrations.llms import MockLLMProvider
    from ragtree.integrations.vectorstores import InMemoryVectorStore
    from ragtree.retrieval import DenseRetriever
    from ragtree.tasks import QuestionAnsweringTask

    store = InMemoryVectorStore()
    store.add_chunks(
        [Chunk(id=f"{d}-c0", document_id=d, text=t) for d, t in _DEMO_CORPUS]
    )
    pipeline = RAGTreePipeline(
        retriever=DenseRetriever(store, top_k=2),
        generator=MockLLMProvider(
            reply="The pump failed because its mechanical seal wore out [maint-001/maint-001-c0]."
        ),
    )
    result = pipeline.run(QuestionAnsweringTask("Why did pump P-102 fail?"))

    console.print("[bold]Question:[/bold] Why did pump P-102 fail?")
    console.print(f"[bold]Answer:[/bold] {result.output}\n")
    table = Table(title="Evidence")
    table.add_column("Reference")
    table.add_column("Score", justify="right")
    table.add_column("Text")
    for span in result.evidence:
        table.add_row(
            f"{span.document_id}/{span.chunk_id}", f"{span.score:.3f}", span.text
        )
    console.print(table)
    console.print(
        "\nSwap MockLLMProvider/InMemoryVectorStore for LiteLLM/Qdrant/Chroma — "
        "the pipeline stays identical (BYOS)."
    )


def _demo_relation_extraction() -> None:
    from ragtree.evaluation.relation_evaluator import RelationEvaluator
    from ragtree.integrations.llms import MockLLMProvider
    from ragtree.tasks import RelationExtractionTask

    doc = _DEMO_RE_DOC
    task = RelationExtractionTask(list(doc["relations"].keys()), document=doc)
    pipeline = RAGTreePipeline(
        generator=MockLLMProvider(reply=json.dumps(doc["relations"])),
        evaluator=RelationEvaluator(),
    )
    result = pipeline.run(task, reference=doc["relations"])

    console.print(f"[bold]Document:[/bold] {doc['document_id']} — {doc['title']}")
    table = Table(title="Extracted relations")
    table.add_column("Type")
    table.add_column("Head")
    table.add_column("Tail")
    for rel_type, pairs in result.output.items():
        for head, tail in pairs:
            table.add_row(rel_type, head, tail)
    console.print(table)
    console.print(
        f"micro P/R/F1 vs gold: {result.metrics['precision']:.2f} / "
        f"{result.metrics['recall']:.2f} / {result.metrics['f1']:.2f}"
    )


def demo(
    which: str = typer.Argument("semantic-rag", help="semantic-rag | relation-extraction")
) -> None:
    """Run a deterministic demo pipeline (no extras, no network)."""
    if which == "semantic-rag":
        _demo_semantic_rag()
    elif which == "relation-extraction":
        _demo_relation_extraction()
    else:
        console.print(f"Unknown demo {which!r}. Try: semantic-rag, relation-extraction")
        raise typer.Exit(code=1)


# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------


def run(
    config: Path = typer.Option(..., "--config", "-c", help="Run config YAML"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Override outputs.directory"
    ),
) -> None:
    """Execute a declarative run config (see examples/configs/)."""
    from ragtree.apps.runner import run_from_config

    summary = run_from_config(config, output_dir=output)
    table = Table(title=f"Run {summary['run_id']}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("task", str(summary["task"]))
    table.add_row("documents", str(summary["n_documents"]))
    table.add_row("results", str(summary["n_results"]))
    table.add_row("output dir", summary["output_dir"])
    for key, value in (summary.get("metrics") or {}).items():
        table.add_row(
            f"metric: {key}", f"{value:.4f}" if isinstance(value, float) else str(value)
        )
    console.print(table)


# ----------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------


def evaluate(
    gold: Path = typer.Option(..., help="Gold JSONL with 'relations' per document"),
    pred: Path = typer.Option(..., help="Predictions JSONL with 'pred_relations'"),
    doc_type: str = typer.Option("all", help="Filter documents by 'type' field"),
    ignore_label: list[str] = typer.Option(
        [], "--ignore-label", help="Relation labels to ignore (repeatable)"
    ),
    output: Optional[Path] = typer.Option(None, help="Write full metrics JSON here"),
) -> None:
    """Evaluate relation predictions with the historical benchmark metrics."""
    from ragtree.evaluation.relations.runner import evaluate_relations

    metrics = evaluate_relations(
        gold_path=gold,
        pred_path=pred,
        doc_type_filter=doc_type,
        ignore_labels=list(ignore_label),
    )
    micro = metrics["micro"]
    counts = metrics["counts"]
    table = Table(title="Relation evaluation (micro)")
    for column in ("precision", "recall", "f1"):
        table.add_column(column, justify="right")
    table.add_row(*(f"{micro[c]:.4f}" for c in ("precision", "recall", "f1")))
    console.print(table)
    console.print(
        f"tp={counts['tp']} fp={counts['fp']} fn={counts['fn']} "
        f"docs={counts['num_docs_eval']} missing_gold={counts['num_docs_missing_gold']}"
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        console.print(f"Full metrics written to {output}")


# ----------------------------------------------------------------------
# export
# ----------------------------------------------------------------------

_EXPORTER_NAMES = ("json", "jsonl", "csv", "graph-csv")


def export(
    input: Path = typer.Option(..., "--input", "-i", help="RAGResult JSON file"),
    format: str = typer.Option("jsonl", "--format", "-f", help="|".join(_EXPORTER_NAMES)),
    output: Path = typer.Option(..., "--output", "-o", help="Destination path"),
) -> None:
    """Re-export a saved RAGResult through any exporter."""
    from ragtree.integrations.exporters import (
        CsvExporter,
        GraphCsvExporter,
        JsonExporter,
        JsonlExporter,
    )

    exporters = {
        "json": JsonExporter,
        "jsonl": JsonlExporter,
        "csv": CsvExporter,
        "graph-csv": GraphCsvExporter,
    }
    if format not in exporters:
        console.print(f"Unknown format {format!r}. Choose from: {', '.join(_EXPORTER_NAMES)}")
        raise typer.Exit(code=1)

    result = RAGResult.model_validate_json(input.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    exporters[format]().export(result, str(output))
    console.print(f"Exported {input} -> {output} ({format})")


# ----------------------------------------------------------------------
# serve / workbench
# ----------------------------------------------------------------------


def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address"),
    port: int = typer.Option(8000, help="Port"),
    reload: bool = typer.Option(False, help="Auto-reload (development)"),
) -> None:
    """Serve the RAGTree FastAPI surface (extra: api)."""
    require_extra("fastapi", "api")
    require_extra("uvicorn", "api")
    import uvicorn

    uvicorn.run("ragtree.apps.api.app:create_app", factory=True, host=host, port=port, reload=reload)


def workbench() -> None:
    """Launch the Streamlit workbench (extra: ui)."""
    require_extra("streamlit", "ui")
    import subprocess
    import sys

    app_path = Path(__file__).resolve().parent.parent / "apps" / "streamlit" / "app.py"
    raise typer.Exit(
        subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)])
    )


__all__ = ["demo", "run", "evaluate", "export", "serve", "workbench"]
