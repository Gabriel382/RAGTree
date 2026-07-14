# ragtree/core/errors.py
"""Exception hierarchy and optional-dependency helpers for RAGTree.

The BYOS rule (design doc, section 7.2): a missing optional dependency must
produce a helpful error naming the pip extra to install, never an
import-time crash.
"""

from __future__ import annotations

__all__ = [
    "RagtreeError",
    "ConfigurationError",
    "MissingDependencyError",
    "require_extra",
]


class RagtreeError(Exception):
    """Base class for all RAGTree errors."""


class ConfigurationError(RagtreeError):
    """Raised when a configuration file or value is missing or invalid."""


class MissingDependencyError(RagtreeError, ImportError):
    """Raised when an optional integration is used without its extra installed."""


def require_extra(package_name: str, extra_name: str) -> None:
    """Assert that an optional dependency is importable.

    Parameters
    ----------
    package_name:
        Import name of the required package (e.g. ``chromadb``).
    extra_name:
        Name of the pip extra that provides it (e.g. ``vector-chroma``).

    Raises
    ------
    MissingDependencyError
        If the package cannot be imported. The message tells the user which
        extra to install, e.g. ``pip install 'ragtree[vector-chroma]'``.
    """
    try:
        __import__(package_name)
    except ImportError as exc:
        raise MissingDependencyError(
            f"This feature requires the '{extra_name}' extra. "
            f"Install it with: pip install 'ragtree[{extra_name}]'"
        ) from exc
