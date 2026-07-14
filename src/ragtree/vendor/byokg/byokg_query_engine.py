# ragtree/vendor/byokg/byokg_query_engine.py
from .graph_retrievers import GraphRetriever

class BYOKGQueryEngine:
    """
    Query engine (QA-oriented) that:
      - retrieves graph context
      - feeds it to an LLM
    """
    def __init__(self, graph_store, llm, embedding_model=None):
        self.retriever = GraphRetriever(graph_store, embedding_model=embedding_model)
        self.llm = llm

    def answer(self, question: str, mentions=None):
        mentions = mentions or [question]
        retrieved = self.retriever.retrieve(mentions)

        prompt = (
            "You are given graph context.\n\n"
            f"GRAPH CONTEXT:\n{retrieved['context']}\n\n"
            f"QUESTION:\n{question}\n\n"
            "Answer:"
        )

        return self.llm.generate(prompt)
