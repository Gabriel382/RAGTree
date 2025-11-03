# scripts/run_preprocess.py
from ragtree.core.config import load_config
from ragtree.utils.seed import set_seed

cfg = load_config()
set_seed(cfg["project"]["seed"])

print(cfg["paths"]["data_raw"])
