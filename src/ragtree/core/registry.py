# ragtree/core/registry.py
_REG = {}

def register(kind: str, name: str):
    def deco(cls):
        _REG.setdefault(kind, {})[name] = cls
        return cls
    return deco

def build(kind: str, name: str, **kwargs):
    return _REG[kind][name](**kwargs)


# Kinds you’ll use:
# "preprocessing.loader", "preprocessing.chunker", "preprocessing.indexer"
# "processing.llm", "processing.retriever", "processing.reranker", "processing.rag"
# "postprocessing.pruner", "postprocessing.exporter", "postprocessing.evaluator"