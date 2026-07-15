"""Streamlit workbench: syntax always checked; real import when installed."""

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.ui]

APP_PATH = (
    Path(__file__).parents[2] / "src" / "ragtree" / "apps" / "streamlit" / "app.py"
)


def test_workbench_source_is_valid_python():
    ast.parse(APP_PATH.read_text(encoding="utf-8"))


def test_workbench_imports_when_streamlit_installed():
    pytest.importorskip("streamlit")
    importlib.import_module("ragtree.apps.streamlit.app")
