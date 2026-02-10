# ragtree/preprocessing/ingest/convert_registry.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Dict, Any, Type
import hashlib


class BaseConverter(ABC):
    """
    Base class for dataset converters.

    Each converter must yield docs with at least:
      - document_id : str  (e.g. "MAVEN_ERE - <hash>")
      - title       : str
      - text        : str
      - type        : str  ("train" | "validation" | "test" | "unknown")
      - sentences   : list[str]
      - tokens      : list[list[str]]
      - entities    : dict[str, Any]   # key: "0", "1", ..., value: entity name
      - relations   : dict[str, list[list[str]]]  # type -> [[head_id, tail_id], ...]
    """

    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)

    @staticmethod
    def _text_hash(text: str, length: int = 16) -> str:
        """Deterministic short hash for document_id/title."""
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        return h[:length]

    @classmethod
    def make_document_id(cls, dataset_name: str, text: str) -> str:
        """Build document_id = '<DATASET> - <hash>'."""
        return f"{dataset_name} - {cls._text_hash(text)}"

    @abstractmethod
    def iter_docs(self) -> Iterator[Dict[str, Any]]:
        """Yield normalized docs."""
        ...


CONVERTERS: Dict[str, Type[BaseConverter]] = {}


def register(name: str):
    def deco(cls: Type[BaseConverter]):
        CONVERTERS[name] = cls
        return cls
    return deco
