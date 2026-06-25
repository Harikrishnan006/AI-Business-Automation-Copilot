import logging
import io

import streamlit as st

from ai_engine import ask_with_context, ask_without_context
from document_processor import extract_pages_from_pdf, create_chunks
from vector_store import store_chunks, get_top_chunks, get_stored_sources, delete_source

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

st.set_page_config(
    page_title="AI Business Automation Copilot",
    page_icon="🤖",
    layout="wide",
)


# ─────────────────────────────────────────────
# SESSION STATE INITIALISATION
#
# st.session_state persists values across reruns
# within the same browser session. We use it for:
#   - chat_history: list of {question, answer, chunks}
#     so the user can scroll back through the session
#
# We initialise once here at the top. If the key
# already exists (page rerun), we leave it alone.
# ─────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ─────────────────────────────────────────────
# CACHED INGESTION
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def process_and_store_file(file_bytes: bytes, filename: str) -> int:
    """Extract, chunk, embed, and store one PDF. Returns chunk count."""
    pages = extract_pages_from_pdf(io.BytesIO(file_bytes))
    if not pages:
        return 0
    chunks = create_chunks(pages, source=filename)
    store_chunks(chunks)
    return len(chunks)


# ─────────────────────────────────────────────
# CONFIDENCE HELPER
# ─────────────────────────────────────────────

def _confidence_label(score: float) -> tuple[str, str]:
    """Return (label, colour) for a confidence score 0–100."""
    if score >= 75:
        return "High", "🟢"
    elif score >= 45:
        return "Medium", "🟡"
    else:
        return "Low", "🔴"


# ─────────────────────────────────────────────
# TAB 1 — UPLOAD
# ─────────────────────────────────────────────

def render_upload_tab() -> None:
    st.subheader("Upload Documents")
    st.caption("Supported format: PDF. Upload SOPs, contracts, FAQs, manuals, policies.")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.info("No files uploaded yet. Drop one or more PDFs above to get started.")
        return

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        with st.spinner(f"Indexing {uploaded_file.name}…"):
            chunk_count = process_and_store_file(file_bytes, uploaded_file.name)

        if chunk_count > 0:
            st.success(f"✅ **{uploaded_file.name}** — {chunk_count} chunks indexed")
        else:
            st.warning(
                f"⚠️ **{uploaded_file.name}** — no text extracted. "
                "The PDF may be scanned or image-based."
            )


# ─────────────────────────────────────────────
# TAB 2 — ASK
# ─────────────────────────────────────────────

def render_ask_tab() -> None:
    sources = get_stored_sources()

    if not sources:
        st.info("No documents indexed yet. Go to the **Upload** tab to add documents first.")
        return

    # ── Controls ──────────────────────────────
    col_scope, col_k = st.columns([3, 1])

    with col_scope:
        search_scope = st.selectbox(
            "Search scope",
            options=["All documents"] + sources,
            help="Restrict answers to one document or search across all.",
        )

    with col_k:
        top_k = st.slider("Chunks", min_value=1, max_value=10, value=5,
                          help="How many document chunks to retrieve per question.")

    source_filter = None if search_scope == "All documents" else search_scope

    # ── Input ─────────────────────────────────
    question = st.chat_input("Ask a question about your documents…")

    # ── Process new question ──────────────────
    if question:
        with st.spinner("Searching knowledge base…"):
            chunks = get_top_chunks(query=question, top_k=top_k, source_filter=source_filter)

        if not chunks:
            st.warning("No relevant chunks found. Try rephrasing or uploading more documents.")
        else:
            with st.spinner("Generating answer…"):
                try:
                    answer = ask_with_context(question, chunks)
                except RuntimeError as e:
                    st.error(f"Failed to generate answer: {e}")
                    answer = None

            if answer:
                # Store in session history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "chunks": chunks,
                })

    # ── Render full chat history ──────────────
    # Shows oldest → newest, just like a real chat UI.
    # Each answer includes confidence and source citations.
    if not st.session_state.chat_history:
        st.markdown(
            "<div style='text-align:center;padding:3rem 0;"
            "color:var(--color-text-tertiary);font-size:14px'>"
            "Ask a question above to get started.</div>",
            unsafe_allow_html=True,
        )
        return

    for entry in st.session_state.chat_history:
        # User bubble
        with st.chat_message("user"):
            st.markdown(entry["question"])

        # Assistant bubble
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])

            # Confidence summary line
            if entry["chunks"]:
                avg_conf = sum(c["confidence"] for c in entry["chunks"]) / len(entry["chunks"])
                label, dot = _confidence_label(avg_conf)
                top_sources = sorted(
                    {c["source"] for c in entry["chunks"]}
                )
                source_str = ", ".join(f"`{s}`" for s in top_sources)
                st.caption(
                    f"{dot} **{label} confidence** ({avg_conf:.0f}/100) · "
                    f"Sources: {source_str}"
                )

            # Source chunks in expander — clean, not a wall of text
            with st.expander("📎 View source chunks", expanded=False):
                for i, chunk in enumerate(entry["chunks"], start=1):
                    conf_label, conf_dot = _confidence_label(chunk["confidence"])
                    st.markdown(
                        f"**Chunk {i}** &nbsp;·&nbsp; "
                        f"`{chunk['source']}` p.{chunk['page']} &nbsp;·&nbsp; "
                        f"{conf_dot} {conf_label} ({chunk['confidence']:.0f})"
                    )
                    st.text(chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else ""))
                    if i < len(entry["chunks"]):
                        st.divider()

    # Clear history button — bottom of chat
    if st.session_state.chat_history:
        if st.button("🗑 Clear conversation", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


# ─────────────────────────────────────────────
# TAB 3 — KNOWLEDGE BASE
# ─────────────────────────────────────────────

def render_knowledge_base_tab() -> None:
    sources = get_stored_sources()

    if not sources:
        st.info("No documents indexed yet. Upload PDFs in the **Upload** tab.")
        return

    st.subheader(f"{len(sources)} document(s) in knowledge base")

    for source in sources:
        col_name, col_btn = st.columns([6, 1])
        with col_name:
            st.markdown(f"📄 &nbsp; {source}")
        with col_btn:
            if st.button("Remove", key=f"del_{source}", type="secondary"):
                delete_source(source)
                process_and_store_file.clear()
                st.rerun()

    st.divider()

    # ── General AI Q&A ────────────────────────
    st.subheader("🌐 General AI Question")
    st.caption(
        "Uses Gemini's general knowledge — **not** your uploaded documents. "
        "Answers may not reflect your company's policies."
    )

    general_q = st.text_area(
        "Ask anything",
        placeholder="e.g. What does RAG stand for?",
        height=80,
        label_visibility="collapsed",
    )

    if st.button("Ask Gemini", type="primary"):
        if not general_q.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking…"):
                try:
                    answer = ask_without_context(general_q)
                    st.markdown("**Response**")
                    st.markdown(answer)
                except RuntimeError as e:
                    st.error(f"Error: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    st.title("🤖 AI Business Automation Copilot")
    st.caption(
        "Ask questions about your company documents. "
        "Answers are grounded in uploaded files with source citations."
    )

    tab_ask, tab_upload, tab_kb = st.tabs([
        "💬 Ask",
        "📂 Upload Documents",
        "🗂 Knowledge Base",
    ])

    with tab_ask:
        render_ask_tab()

    with tab_upload:
        render_upload_tab()

    with tab_kb:
        render_knowledge_base_tab()


if __name__ == "__main__":
    main()
