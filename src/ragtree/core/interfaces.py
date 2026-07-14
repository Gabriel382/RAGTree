# ragtree/core/interfaces.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .types import Query, Evidence, CausalTree

# —— Phase roots ——
class Preprocessor(ABC):
    @abstractmethod
    def run(self, situation_dir: str, **kw) -> Dict[str, Any]: ...

class Processor(ABC):
    @abstractmethod
    def run(self, query: Query, **kw) -> List[CausalTree]: ...

class Postprocessor(ABC):
    @abstractmethod
    def run(self, trees: List[CausalTree], **kw) -> List[CausalTree]: ...

# —— Processing sub-ABCs ——
class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kw) -> str: ...
    @abstractmethod
    def name(self) -> str: ...

class Retriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int, filters: Dict[str, Any] | None = None) -> List[Evidence]: ...

class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, evidences: List[Evidence], top_k: int) -> List[Evidence]: ...

class RAGStrategy(Processor):  # inherits Processor
    def __init__(self, llm: LLMClient, retriever: Retriever, reranker: Reranker | None = None):
        self.llm, self.retriever, self.reranker = llm, retriever, reranker
