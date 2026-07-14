# ragtree/vendor/byokg/embedding.py
from abc import ABC, abstractmethod

class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.
    """
    @abstractmethod
    def embed(self, text):
        """
        Generate embedding for the given text.
        :param text: Input string
        :return: Vector embedding
        """
        pass
