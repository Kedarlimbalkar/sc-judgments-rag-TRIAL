"""
Supreme Court of India - AI Judgments RAG Assistant
Clean, Sober Gemini-Style Streamlit Web Application for Google Cloud Run.
Fully Dynamic RAG Retrieval & Streaming Generation.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional
import streamlit as st

# Configure Streamlit page layout
st.set_page_config(
    page_title="Supreme Court of India — AI Legal Assistant",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Sober Gemini-Style UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #131314;
        color: #e3e3e3;
    }
    
    /* Header Container */
    .gemini-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1.1rem 1.25rem;
        background: #1e1f20;
        border: 1px solid #333538;
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    
    .gemini-logo {
        font-size: 2.1rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.3rem 0.6rem;
    }
    
    .gemini-title {
        font-family: 'Google Sans', sans-serif;
        font-size: 1.35rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0;
    }
    
    .gemini-subtitle {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 0.15rem;
    }
    
    /* Citations */
    .citation-tag {
        display: inline-block;
        background: #282a2c;
        border: 1px solid #3c4043;
        color: #8ab4f8;
        padding: 0.25rem 0.65rem;
        border-radius: 8px;
        font-size: 0.76rem;
        margin: 0.2rem;
    }
    
    /* Ground Truth Judicial Cards */
    .parent-card {
        background: #1e1f20;
        border: 1px solid #333538;
        border-radius: 10px;
        padding: 0.85rem;
        margin-top: 0.5rem;
        font-size: 0.82rem;
    }
    
    .parent-card-header {
        display: flex;
        justify-content: space-between;
        color: #9ca3af;
        font-size: 0.75rem;
        margin-bottom: 0.35rem;
    }

    /* Style Streamlit Chat Input */
    .stChatInputContainer {
        border-color: #333538 !important;
        background-color: #1e1f20 !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Backend Service Resolver
# ------------------------------------------------------------------------------
@st.cache_resource
def get_backend_services():
    """Initializes database pool, Parquet vector store, and Vertex AI service manager."""
    from online_app import services, RetrievalEngine, CitationVerificationGate, ParquetVectorStore
    try:
        services.initialize()
    except Exception:
        pass

    parquet_store = ParquetVectorStore()
    retrieval = RetrievalEngine(services, parquet_store=parquet_store)
    gate = CitationVerificationGate(services)
    return {
        "services": services,
        "retrieval": retrieval,
        "gate": gate,
        "parquet_store": parquet_store
    }

backend = get_backend_services()

# ------------------------------------------------------------------------------
# Minimal Distraction-Free Sidebar
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚖️ SC Judgments")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown(
        "<div style='color: #9ca3af; font-size: 0.8rem;'>"
        "Supreme Court of India AI Legal Research Assistant.<br>"
        "Queries are synthesized and verified dynamically against original judicial records."
        "</div>",
        unsafe_allow_html=True
    )

# ------------------------------------------------------------------------------
# Clean Gemini-Style Header
# ------------------------------------------------------------------------------
st.markdown("""
<div class="gemini-header">
    <div class="gemini-logo">⚖️</div>
    <div>
        <h1 class="gemini-title">Supreme Court of India</h1>
        <div class="gemini-subtitle">
            AI Legal Assistant • Dynamic Precedent & Jurisprudence Retrieval
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Chat History Management
# ------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Initial Welcome & Suggested Prompts if conversation is empty
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 1rem 1rem 1rem;">
        <h2 style="font-family: 'Google Sans', sans-serif; font-weight: 500; font-size: 1.5rem; color: #e3e3e3; margin-bottom: 0.4rem;">
            How can I assist your legal research today?
        </h2>
        <p style="color: #9ca3af; font-size: 0.9rem; margin-bottom: 1.25rem;">
            Ask any question regarding Supreme Court judgments, parties, constitutional articles, or precedents.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚖️ Shreekantiah Ramayya Munipalli Case Rationale", use_container_width=True):
            st.session_state.pending_query = "Who was Shreekantiah Ramayya Munipalli and what was the Supreme Court's holding in this case?"
            st.rerun()
        if st.button("🏛️ Twin Conditions for Bail under Section 45 PMLA", use_container_width=True):
            st.session_state.pending_query = "What are the twin conditions for grant of bail under Section 45 of PMLA?"
            st.rerun()
    with col2:
        if st.button("📜 Article 21 Right to Speedy Trial Principles", use_container_width=True):
            st.session_state.pending_query = "Article 21 right to speedy trial and undertrial detention principles"
            st.rerun()
        if st.button("📑 Arbitrator Appointment under Section 11", use_container_width=True):
            st.session_state.pending_query = "What principles govern appointment of arbitrators under Section 11?"
            st.rerun()

# Display Chat History Messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="⚖️"):
            st.markdown(msg["content"])
            
            citations = msg.get("citations", [])
            if citations:
                chips_html = "".join([f'<span class="citation-tag">📌 {c}</span>' for c in citations])
                st.markdown(f'<div style="margin-top: 0.6rem; margin-bottom: 0.4rem;">{chips_html}</div>', unsafe_allow_html=True)
                
            parents = msg.get("parents", [])
            if parents:
                with st.expander(f"📚 View Reference Passages ({len(parents)})"):
                    for p in parents:
                        p_dict = p if isinstance(p, dict) else (p.model_dump() if hasattr(p, "model_dump") else p.dict())
                        st.markdown(f"""
                        <div class="parent-card">
                            <div class="parent-card-header">
                                <span><strong>{p_dict.get('section_title', 'Judicial Section')}</strong></span>
                                <span>Pages {p_dict.get('page_start', '?')}-{p_dict.get('page_end', '?')} • Case: {p_dict.get('case_number', 'SC')}</span>
                            </div>
                            <div style="font-weight: 600; color: #fff; margin-bottom: 0.25rem;">
                                {p_dict.get('case_title', 'Supreme Court of India')} ({p_dict.get('judgment_date', 'N/A')})
                            </div>
                            <p style="color: #c4c7c5; line-height: 1.5; font-family: serif;">
                                "{p_dict.get('content', '')[:400]}..."
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Free-Form UI Input Flow & Dynamic Streaming Generation
# ------------------------------------------------------------------------------
prompt = st.chat_input("Ask a question about Supreme Court judgments, statutes, or constitutional law...")

# Handle Pending Query from Suggested Prompt Buttons
if hasattr(st.session_state, "pending_query"):
    prompt = st.session_state.pending_query
    del st.session_state.pending_query

if prompt:
    # 1. Render User Message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Dynamic Vector Search Across All Stored Chunks & Streaming Assistant Response
    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("Searching Supreme Court records & analyzing legal precedents..."):
            retriever = backend["retrieval"]
            gate = backend["gate"]

            # Perform dynamic vector/similarity search across all chunks (fetching top 8-10 chunks)
            chunks = retriever.hybrid_search_rrf(prompt, top_k=8)
            parent_ids = list(dict.fromkeys(c.get("parent_id") or c.get("chunk_id") for c in chunks))
            parents = retriever.expand_parents(parent_ids)

        # Stream generated response token-by-token cleanly into chat interface
        stream_gen = gate.stream_synthesize(prompt, parents)
        full_response = st.write_stream(stream_gen)

        # Grounded Citations
        citations = []
        for p in parents:
            title = p.case_title if hasattr(p, "case_title") else p.get("case_title", "")
            cite = p.citation if hasattr(p, "citation") else p.get("citation", "")
            if title:
                tag = f"{title} ({cite})" if cite else title
                if tag not in citations:
                    citations.append(tag)

        if citations:
            chips_html = "".join([f'<span class="citation-tag">📌 {c}</span>' for c in citations[:4]])
            st.markdown(f'<div style="margin-top: 0.6rem; margin-bottom: 0.4rem;">{chips_html}</div>', unsafe_allow_html=True)

        if parents:
            with st.expander(f"📚 View Reference Passages ({len(parents)})"):
                for p in parents:
                    p_dict = p if isinstance(p, dict) else (p.model_dump() if hasattr(p, "model_dump") else p.dict())
                    st.markdown(f"""
                    <div class="parent-card">
                        <div class="parent-card-header">
                            <span><strong>{p_dict.get('section_title', 'Judicial Section')}</strong></span>
                            <span>Pages {p_dict.get('page_start', '?')}-{p_dict.get('page_end', '?')} • Case: {p_dict.get('case_number', 'SC')}</span>
                        </div>
                        <div style="font-weight: 600; color: #fff; margin-bottom: 0.25rem;">
                            {p_dict.get('case_title', 'Supreme Court of India')} ({p_dict.get('judgment_date', 'N/A')})
                        </div>
                        <p style="color: #c4c7c5; line-height: 1.5; font-family: serif;">
                            "{p_dict.get('content', '')[:400]}..."
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

    # 3. Save Assistant Message in Session State
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "citations": citations[:4],
        "parents": [p.model_dump() if hasattr(p, "model_dump") else (p if isinstance(p, dict) else p.dict()) for p in parents]
    })

# ------------------------------------------------------------------------------
# Entrypoint Support for Direct Python Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    from streamlit.runtime import exists
    if not exists():
        from streamlit.web import cli as stcli
        port = os.environ.get("PORT", "8080")
        sys.argv = [
            "streamlit",
            "run",
            os.path.abspath(__file__),
            f"--server.port={port}",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.gatherUsageStats=false"
        ]
        sys.exit(stcli.main())
