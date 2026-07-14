# ragtree/vendor/byokg/index.py
from abc import ABC, abstractmethod

class Index(ABC):
    """
    Abstract base class for indexes.
    """
    @abstractmethod
    def build(self, items):
        pass

    @abstractmethod
    def query(self, q, top_k=5):
        pass
