"""Unit tests for the error hierarchy and require_extra helper."""

import pytest

from ragtree.core.errors import MissingDependencyError, RagtreeError, require_extra


def test_require_extra_passes_for_available_package():
    require_extra("json", "anything")  # stdlib import always available


def test_require_extra_raises_helpful_error():
    with pytest.raises(MissingDependencyError) as excinfo:
        require_extra("surely_not_an_installed_package_42", "vector-chroma")
    assert "ragtree[vector-chroma]" in str(excinfo.value)


def test_missing_dependency_error_is_import_error_and_ragtree_error():
    assert issubclass(MissingDependencyError, ImportError)
    assert issubclass(MissingDependencyError, RagtreeError)
