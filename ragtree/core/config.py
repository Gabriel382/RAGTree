from pathlib import Path
import yaml

DEFAULT_CFG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CFG
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # resolve relative paths based on project root
    root = DEFAULT_CFG.parents[1]  # points to .../ragtree/
    for key, value in cfg.get("paths", {}).items():
        cfg["paths"][key] = str((root / value).resolve())

    return cfg
