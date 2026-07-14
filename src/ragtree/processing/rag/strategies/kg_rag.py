# ragtree/processing/rag/strategies/kg_rag.py
from ragtree.core.interfaces import IRAGStrategy

class KGRAG(IRAGStrategy):
    def __init__(self, retriever, llm, kg_client):
        self.retriever = retriever
        self.llm = llm
        self.kg = kg_client

    def run(self, query: str, k: int = 5) -> dict:
        ctx = self.retriever.retrieve(query, k=k)
        kg_triples = self.kg.lookup(query, k=k)
        prompt = (
            "You are a causality extractor.\n"
            "Use the following text context and KG triples to extract cause-effect pairs.\n"
            f"TEXT:\n{[c['text'] for c in ctx]}\n"
            f"KG:\n{kg_triples}\n"
            "Return JSON array."
        )
        answer = self.llm.generate(prompt)
        return {"answer": answer, "text_ctx": ctx, "kg_ctx": kg_triples}
