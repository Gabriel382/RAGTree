"""RAGTree BYOS demo — semantic RAG with ZERO optional extras.

Run from the repo root:

    python examples/semantic_rag_demo.py

Swapping the stack is the whole point (design doc, section 17.3):

    generator  = LiteLLMProvider(model="openai/gpt-4o-mini")   # pip install "ragtree[llm-litellm]"
    store      = QdrantVectorStore(embedder=..., url=...)      # pip install "ragtree[vector-qdrant]"
    store      = ChromaVectorStore(persist_directory=".chroma") # pip install "ragtree[vector-chroma]"

The pipeline code below stays identical.
"""

from ragtree import RAGTreePipeline
from ragtree.core.schemas import Chunk
from ragtree.integrations.llms import MockLLMProvider
from ragtree.integrations.vectorstores import InMemoryVectorStore
from ragtree.retrieval import DenseRetriever
from ragtree.tasks import QuestionAnsweringTask

CORPUS = [
    ("maint-001", "Pump P-102 failed on 12 March because the mechanical seal wore out."),
    ("maint-002", "Routine maintenance was performed in June on all centrifugal pumps."),
    ("maint-003", "Alarm 7741 was triggered by a pressure spike in line L-3."),
]


def main() -> None:
    store = InMemoryVectorStore()
    store.add_chunks(
        [Chunk(id=f"{doc_id}-c0", document_id=doc_id, text=text) for doc_id, text in CORPUS]
    )

    pipeline = RAGTreePipeline(
        retriever=DenseRetriever(store, top_k=2),
        generator=MockLLMProvider(
            reply="The pump failed because its mechanical seal wore out [maint-001/maint-001-c0]."
        ),
    )

    result = pipeline.run(QuestionAnsweringTask("Why did pump P-102 fail?"))

    print(f"Answer:  {result.output}\n")
    print("Evidence:")
    for span in result.evidence:
        print(f"  [{span.document_id}/{span.chunk_id}] score={span.score:.3f} {span.text}")


if __name__ == "__main__":
    main()
