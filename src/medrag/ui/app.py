"""Streamlit demo UI for medrag.

Run with: streamlit run src/medrag/ui/app.py
"""

from __future__ import annotations

import time

import streamlit as st

from medrag.agent.graph import MedRagAgent
from medrag.config import MAX_QUESTION_CHARS, get_settings
from medrag.ratelimit import client_from_forwarded, get_limiter

st.set_page_config(
    page_title="MedRAG — GLP-1 Cardiovascular Literature Agent",
    page_icon="💊",
    layout="wide",
)

st.title("MedRAG")
st.caption(
    "Agentic RAG over 498 recent PubMed papers on GLP-1 receptor agonists and "
    "cardiovascular outcomes, 278 of them with full text from PubMed Central. "
    "Built for medical affairs literature review."
)

with st.sidebar:
    st.header("About")
    st.markdown(
        "**Stack:** PubMed/PMC · bge embeddings · Pinecone · BM25 + RRF · "
        "bge-reranker · LangGraph · DeepSeek V4.1 Flash (OpenRouter)\n\n"
        "**Eval:** Recall@10, nDCG@10, RAGAS faithfulness on 33 held-out questions.\n\n"
        "[Source on GitHub](https://github.com/SachiK0thari/medrag)"
    )
    _settings = get_settings()
    if _settings.rate_limit_per_client > 0:
        st.caption(
            f"Public demo: limited to {_settings.rate_limit_per_client} questions per visitor "
            f"every {_settings.rate_limit_window_s // 60} minutes to keep LLM costs bounded."
        )
    st.divider()
    st.subheader("Example questions")
    examples = [
        "What was the primary cardiovascular outcome in the LEADER trial?",
        "How does semaglutide compare to dulaglutide for MACE reduction?",
        "What are the proposed mechanisms of GLP-1 RA cardiovascular benefit?",
        "How much does semaglutide lower systolic blood pressure in cardiovascular outcome trials?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["question"] = ex


@st.cache_resource
def load_agent() -> MedRagAgent:
    return MedRagAgent()


question = st.text_input(
    "Clinical question",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What did the SELECT trial show about semaglutide in non-diabetic obesity?",
    max_chars=MAX_QUESTION_CHARS,
)


def _check_rate_limit() -> None:
    """Count this question against the limits; show why and stop if refused."""
    # HF Spaces (and any reverse proxy) puts the visitor's address in
    # X-Forwarded-For; without the header (local runs) everyone shares one key.
    client = client_from_forwarded(st.context.headers.get("X-Forwarded-For"))
    decision = get_limiter().acquire(client)
    if not decision.allowed:
        st.warning(decision.message)
        st.stop()


if st.button("Search", type="primary", disabled=not question):
    _check_rate_limit()
    with st.spinner("Running agentic retrieval..."):
        t0 = time.perf_counter()
        agent = load_agent()
        answer = agent.answer(question)
        latency_s = time.perf_counter() - t0

    col_main, col_meta = st.columns([3, 1])
    with col_main:
        st.subheader("Answer")
        st.markdown(answer.text)
        st.subheader("Sources")
        for cit in answer.citations:
            with st.expander(f"{cit.marker} {cit.display_citation}"):
                st.markdown(f"**PMID:** {cit.pmid}")
                st.markdown(f"> {cit.snippet}")
    with col_meta:
        st.metric("Latency", f"{latency_s:.2f}s")
        st.metric("Confidence", f"{answer.confidence:.0%}")
        st.metric("Sub-queries", len(answer.sub_queries))
        st.metric("Chunks considered", answer.total_chunks_considered)
