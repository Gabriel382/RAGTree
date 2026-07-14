# ragtree/processing/orchestrators/pipeline.py
from ...core.types import Query, CausalTree
from ...core.registry import build

def run_ragtree(situation_dir: str, config) -> list[CausalTree]:
    # PRE
    loader = build("preprocessing.loader", config.pre.loader.name, **config.pre.loader.params)
    docs = loader.run(situation_dir)
    chunker = build("preprocessing.chunker", config.pre.chunker.name, **config.pre.chunker.params)
    chunks = chunker.run(docs)
    indexer = build("preprocessing.indexer", config.pre.indexer.name, **config.pre.indexer.params)
    indexer.run(chunks)  # builds/updates BM25 + dense stores

    # PROC
    llm = build("processing.llm", config.proc.llm.name, **config.proc.llm.params)
    retr = build("processing.retriever", config.proc.retriever.name, **config.proc.retriever.params)
    rer  = build("processing.reranker", config.proc.reranker.name, **config.proc.reranker.params) if config.proc.reranker else None
    rag  = build("processing.rag", config.proc.strategy.name, llm=llm, retriever=retr, reranker=rer, **config.proc.strategy.params)

    query = Query(situation_id=config.meta.situation_id, question=config.proc.task.question)
    trees = rag.run(query)

    # POST
    pruner = build("postprocessing.pruner", config.post.pruner.name, **config.post.pruner.params)
    trees = pruner.run(trees)
    exporter = build("postprocessing.exporter", config.post.exporter.name, **config.post.exporter.params)
    exporter.run(trees)
    return trees
