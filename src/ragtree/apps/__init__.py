# ragtree/apps/__init__.py
"""Application surfaces: config runner, FastAPI service, Streamlit workbench.

Surfaces consume the core through protocols; FastAPI and Streamlit modules
import their frameworks at module level and are only imported after
``require_extra`` checks (CLI) or ``importorskip`` (tests).
"""
