# scripts/setup_dirs.py
from pathlib import Path
from ragtree.core.config import load_config
from ragtree.utils.logger import get_logger
from ragtree.utils.seed import set_seed

def main():
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    logger = get_logger("setup_dirs", cfg["project"]["log_dir"])

    # collect all path-like entries
    path_keys = ["data_raw", "data_processed", "models", "kg", "results", "jsonresults"]
    for key in path_keys:
        p = Path(cfg["paths"][key])
        p.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured folder: %s", p.resolve())

    logger.info("All folders created.")

if __name__ == "__main__":
    main()
