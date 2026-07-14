# ragtree/utils/logger.py
import logging
from pathlib import Path
from datetime import datetime

def get_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)

    # console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fh_path = Path(log_dir) / f"{name}_{ts}.log"
    fh = logging.FileHandler(fh_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
