# ragtree/__init__.py
"""RAGTree: a bring-your-own-stack framework for Semantic RAG pipelines."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from ragtree.core import (
    Chunk,
    ConfigurationError,
    Document,
    EvaluationResult,
    EvidenceSpan,
    MissingDependencyError,
    RagtreeError,
    RAGResult,
    RAGTask,
    RAGTreePipeline,
    RelationPrediction,
    RunManifest,
    load_config,
    require_extra,
)

try:
    __version__ = _pkg_version("ragtree")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "Document",
    "Chunk",
    "EvidenceSpan",
    "RAGTask",
    "RAGResult",
    "RelationPrediction",
    "RunManifest",
    "EvaluationResult",
    "RAGTreePipeline",
    "load_config",
    "require_extra",
    "RagtreeError",
    "ConfigurationError",
    "MissingDependencyError",
]
