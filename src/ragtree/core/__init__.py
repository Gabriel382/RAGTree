# ragtree/core/__init__.py
"""RAGTree core: stable schemas, protocols, configuration, registry and errors.

This package must never import optional integration SDKs (design rule,
section 4.1). Everything importable from here works on a bare
``pip install ragtree``.
"""

from .config import load_config
from .pipeline import RAGTreePipeline
from .errors import (
    ConfigurationError,
    MissingDependencyError,
    RagtreeError,
    require_extra,
)
from .protocols import (
    Embedder,
    Evaluator,
    Exporter,
    GraphStore,
    LLMProvider,
    OntologyStore,
    Retriever,
    VectorStore,
)
from .registry import build, register
from .schemas import (
    Chunk,
    Document,
    EvaluationResult,
    EvidenceSpan,
    RAGResult,
    RAGTask,
    RelationPrediction,
    RunManifest,
)

__all__ = [
    # schemas
    "Document",
    "Chunk",
    "EvidenceSpan",
    "RAGTask",
    "RAGResult",
    "RelationPrediction",
    "RunManifest",
    "EvaluationResult",
    # protocols
    "LLMProvider",
    "Embedder",
    "VectorStore",
    "Retriever",
    "GraphStore",
    "OntologyStore",
    "Evaluator",
    "Exporter",
    # pipeline / config / registry / errors
    "RAGTreePipeline",
    "load_config",
    "register",
    "build",
    "RagtreeError",
    "ConfigurationError",
    "MissingDependencyError",
    "require_extra",
]
