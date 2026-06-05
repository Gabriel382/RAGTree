# ragtree/vendor/byokg/utils.py
import re

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s
