# ragtree/datasets/base.py
from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any

class BaseDataset(ABC):
    name: str = "base"

    @abstractmethod
    def load(self) -> Iterable[Dict[str, Any]]:
        """
        Should yield dicts like:
        {
          "doc_id": str,
          "text": str,
          "entities": [...],   # optional
          "relations": [...]   # optional
        }
        """
        ...
