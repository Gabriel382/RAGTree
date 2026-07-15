# ragtree/apps/streamlit/app.py
"""RAGTree Streamlit workbench: interactive QA over a pasted corpus.

Launch with ``ragtree workbench`` (extra: ui). Demonstrates the BYOS swap:
choose the mock provider (offline) or LiteLLM/Ollama when installed.
"""

from __future__ import annotations

import streamlit as st

from ragtree import __version__
from ragtree.apps.runner import build_provider, build_retriever
from ragtree.core.pipeline import RAGTreePipeline
from ragtree.tasks import QuestionAnsweringTask

DEFAULT_CORPUS = """maint-001 | Pump P-102 failed on 12 March because the mechanical seal wore out.
maint-002 | Routine maintenance was performed in June on all centrifugal pumps.
maint-003 | Alarm 7741 was triggered by a pressure spike in line L-3."""

st.set_page_config(page_title="RAGTree workbench", page_icon="🌳", layout="wide")
st.title("🌳 RAGTree workbench")
st.caption(f"Bring-your-own-stack Semantic RAG — v{__version__}")

with st.sidebar:
    st.header("Stack")
    provider_name = st.selectbox("LLM provider", ["mock", "litellm", "ollama"], index=0)
    model = st.text_input(
        "Model", value="", help="LiteLLM/Ollama model id; empty uses the default"
    )
    mock_reply = st.text_area(
        "Mock reply (mock provider only)",
        value="The pump failed because its mechanical seal wore out [maint-001/maint-001-c0].",
    )
    top_k = st.slider("Evidence top_k", 1, 10, 3)

corpus_text = st.text_area("Corpus (one document per line: id | text)", DEFAULT_CORPUS, height=140)
question = st.text_input("Question", "Why did pump P-102 fail?")

if st.button("Run pipeline", type="primary"):
    documents = []
    for line in corpus_text.splitlines():
        if "|" in line:
            doc_id, text = line.split("|", 1)
            documents.append({"id": doc_id.strip(), "text": text.strip()})
    if not documents or not question.strip():
        st.error("Provide at least one 'id | text' document and a question.")
        st.stop()

    spec = {"provider": provider_name, "model": model or None, "reply": mock_reply}
    try:
        provider = build_provider(spec)
    except Exception as exc:  # MissingDependencyError -> actionable message
        st.error(str(exc))
        st.stop()

    pipeline = RAGTreePipeline(
        retriever=build_retriever(documents, {"top_k": top_k}), generator=provider
    )
    result = pipeline.run(QuestionAnsweringTask(question))

    st.subheader("Answer")
    st.write(result.output)

    st.subheader("Evidence")
    st.dataframe(
        [
            {
                "reference": f"{s.document_id}/{s.chunk_id}",
                "score": s.score,
                "text": s.text,
            }
            for s in result.evidence
        ],
        use_container_width=True,
    )

    with st.expander("Raw RAGResult"):
        st.json(result.model_dump(mode="json"))
