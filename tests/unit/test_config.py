"""Unit tests for config loading and path resolution."""

import pytest
import yaml

from ragtree.core.config import load_config
from ragtree.core.errors import ConfigurationError


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_paths_resolve_relative_to_configs_parent(tmp_path):
    cfg_file = tmp_path / "configs" / "default.yaml"
    _write(cfg_file, {"paths": {"data_raw": "data/raw"}, "project": {"seed": 1}})
    cfg = load_config(str(cfg_file))
    assert cfg["paths"]["data_raw"] == str((tmp_path / "data" / "raw").resolve())
    assert cfg["project"]["seed"] == 1


def test_paths_resolve_relative_to_file_dir_outside_configs(tmp_path):
    cfg_file = tmp_path / "my.yaml"
    _write(cfg_file, {"paths": {"out": "results"}})
    cfg = load_config(str(cfg_file))
    assert cfg["paths"]["out"] == str((tmp_path / "results").resolve())


def test_config_without_paths_section_loads(tmp_path):
    cfg_file = tmp_path / "bare.yaml"
    _write(cfg_file, {"project": {"name": "bare"}})
    assert load_config(str(cfg_file))["project"]["name"] == "bare"


def test_missing_explicit_path_raises(tmp_path):
    with pytest.raises(ConfigurationError):
        load_config(str(tmp_path / "nope.yaml"))


def test_default_config_discovered_upward_from_cwd(tmp_path, monkeypatch):
    _write(tmp_path / "configs" / "default.yaml", {"project": {"name": "found"}})
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.delenv("RAGTREE_CONFIG", raising=False)
    assert load_config()["project"]["name"] == "found"


def test_env_var_selects_config(tmp_path, monkeypatch):
    cfg_file = tmp_path / "custom.yaml"
    _write(cfg_file, {"project": {"name": "from-env"}})
    monkeypatch.setenv("RAGTREE_CONFIG", str(cfg_file))
    assert load_config()["project"]["name"] == "from-env"


def test_env_var_pointing_nowhere_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGTREE_CONFIG", str(tmp_path / "ghost.yaml"))
    with pytest.raises(ConfigurationError):
        load_config()
