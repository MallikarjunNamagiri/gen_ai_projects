import streamlit as st
from src.config import (
    DATA_DIR,
    GROQ_API_KEY,
    GROQ_MODEL,
    SEMANTIC_SIMILARITY_THRESHOLD,
    GUARDRAIL_CONFIDENCE_THRESHOLD,
    AUTO_REFRESH_ENABLED,
    AUTO_REFRESH_SECONDS,
)
from src.core.cache import (
    check_semantic_cache,
    store_semantic_cache,
    load_query_json,
)
from src.core.guardrails import rerank_documents_with_guardrail
from src.services.ingestion import (
    calculate_file_hash,
    load_document_into_vectorstore,
    save_and_build_vectorstore,
)
from src.services.rag import create_rag_chain
from src.services.router import (
    contextualize_question,
    is_global_summary_query,
    get_document_outline_context,
    is_greeting_query,
)
from src.ui.styles import inject_custom_css, inject_scroll_button
from src.ui.components import render_assistant_response

# ============================================================
# PAGE CONFIGURATION & STYLES
# ============================================================
st.set_page_config(
    page_title="Document AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "document_hash" not in st.session_state:
    st.session_state.document_hash = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ============================================================
# AUTO REFRESH (OPTIONAL)
# ============================================================
if AUTO_REFRESH_ENABLED:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(
            interval=AUTO_REFRESH_SECONDS * 1000,
            key="document_ai_auto_refresh",
        )
    except ImportError:
        pass

# ============================================================
# SIDEBAR NAVIGATION & CONTROLS
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">✦</div>
            <div>
                <div class="brand-title">Document AI</div>
                <div class="brand-subtitle">Intelligent Document Analysis</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="side-section-title">📂 &nbsp; Document Source</div>',
        unsafe_allow_html=True,
    )

    # Scan persistent /data directory for existing files
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing_files = [
        f.name for f in DATA_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in [".pdf", ".txt"]
    ]

    selected_file = st.selectbox(
        "Select Document",
        options=["-- Upload New File --"] + existing_files,
        help="Choose a previously uploaded file or upload a new one.",
        label_visibility="collapsed",
    )

    # 1. Load an existing file chosen from the dropdown
    if selected_file != "-- Upload New File --":
        file_path = DATA_DIR / selected_file
        with open(file_path, "rb") as f:
            current_hash = calculate_file_hash(f.read())

        if current_hash != st.session_state.document_hash:
            with st.spinner(f"Loading {selected_file}..."):
                vs, chunks, name, fhash = load_document_into_vectorstore(file_path)
                st.session_state.vectorstore = vs
                st.session_state.document_chunks = chunks
                st.session_state.document_name = name
                st.session_state.document_hash = fhash
                st.session_state.rag_chain = None
                st.session_state.messages = []
                st.rerun()

    # 2. Provide the upload widget when "-- Upload New File --" is selected
    else:
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["pdf", "txt"],
            help="Upload a PDF or TXT document to build the semantic index.",
            label_visibility="collapsed",
        )
        if uploaded_file:
            current_hash = calculate_file_hash(uploaded_file.getvalue())
            if current_hash != st.session_state.document_hash:
                with st.spinner("Indexing document..."):
                    vs, chunks, name, fhash = save_and_build_vectorstore(uploaded_file)
                    st.session_state.vectorstore = vs
                    st.session_state.document_chunks = chunks
                    st.session_state.document_name = name
                    st.session_state.document_hash = fhash
                    st.session_state.rag_chain = None
                    st.session_state.messages = []
                    st.rerun()

    if st.session_state.document_name:
        st.markdown(
            f"""
            <div class="active-file-box">
                <div class="active-file-name">📄 {st.session_state.document_name}</div>
                <div class="active-file-status">● Ready for queries</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Telemetry and Pipeline Status Card
    groq_pill = (
        '<span class="pill pill-green">● Connected</span>'
        if GROQ_API_KEY
        else '<span class="pill pill-amber">● Missing Key</span>'
    )
    index_pill = (
        '<span class="pill pill-green">● Ready</span>'
        if st.session_state.vectorstore
        else '<span class="pill pill-gray">● Waiting File</span>'
    )
    cached_doc_count = sum(
        1
        for item in load_query_json().values()
        if item.get("doc_hash") == st.session_state.document_hash
    )

    st.markdown(
        f"""
        <div class="side-section-title">⚡ &nbsp; Engine & Status</div>
        <div class="pipeline-card">
            <div class="pipeline-item">
                <span class="pipeline-label">LLM Provider</span>
                <span class="pipeline-value">Groq {groq_pill}</span>
            </div>
            <div class="pipeline-item">
                <span class="pipeline-label">Model</span>
                <span class="pipeline-value">{GROQ_MODEL}</span>
            </div>
            <div class="pipeline-item">
                <span class="pipeline-label">Doc Vector DB</span>
                <span class="pipeline-value">FAISS {index_pill}</span>
            </div>
            <div class="pipeline-item">
                <span class="pipeline-label">Retrieval</span>
                <span class="pipeline-value">Hybrid + Router</span>
            </div>
            <div class="pipeline-item">
                <span class="pipeline-label">Guardrail</span>
                <span class="pipeline-value">Cutoff ({GUARDRAIL_CONFIDENCE_THRESHOLD})</span>
            </div>
            <div class="pipeline-item">
                <span class="pipeline-label">Semantic Cache</span>
                <span class="pipeline-value">{cached_doc_count} cached</span>
            </div>
            <div class="pipeline-item">
                <span class="pipeline-label">Match Threshold</span>
                <span class="pipeline-value">{int(SEMANTIC_SIMILARITY_THRESHOLD * 100)}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# MAIN APPLICATION HEADER
# ============================================================
head_col1, head_col2 = st.columns([6, 1])
with head_col1:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">AI Document Intelligence</div>
            <h1 class="hero-title">Ask your documents. <span class="gold">Get intelligent answers.</span></h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
with head_col2:
    if st.session_state.messages:
        st.write("")
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()

# ============================================================
# API KEY & DOCUMENT VALIDATION
# ============================================================
if not GROQ_API_KEY:
    st.error("Groq API key not found. Please set GROQ_API_KEY in your .env file.")
    st.stop()

if st.session_state.vectorstore is None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">✦</div>
            <div class="empty-title">No Document Loaded</div>
            <div class="empty-description">
                Upload or select a PDF/TXT document in the sidebar to begin.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Build the hybrid retrieval chain once per document session
if st.session_state.rag_chain is None and st.session_state.document_chunks:
    st.session_state.rag_chain = create_rag_chain(
        st.session_state.vectorstore,
        st.session_state.document_chunks,
    )

prompt, retriever, llm = st.session_state.rag_chain

# ============================================================
# RENDER CHAT HISTORY
# ============================================================
total_msgs = len(st.session_state.messages)
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(
            f"""
            <div class="chat-user">
                <div class="chat-user-bubble">{message["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        render_assistant_response(
            content=message["content"],
            cached=message.get("cached", False),
            score=message.get("score", 1.0),
            sources=message.get("sources", []),
            is_latest=(i == total_msgs - 1),
            msg_idx=i,
        )

# ============================================================
# CHAT INPUT & QUERY PIPELINE
# ============================================================
user_input = st.chat_input("Ask a question about your document...")
active_question = user_input or st.session_state.pending_question

if active_question:
    st.session_state.pending_question = None
    st.session_state.messages.append({"role": "user", "content": active_question})
    st.markdown(
        f"""
        <div class="chat-user">
            <div class="chat-user-bubble">{active_question}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Handle Conversational Greetings (Zero-latency fast-path)
    if is_greeting_query(active_question):
        doc_name = st.session_state.document_name or "your document"
        greeting_response = (
            f"Hello! I am ready to help you analyze **{doc_name}**.\n\n"
            "You can ask me to summarize key sections, explain specific concepts, "
            "or locate detailed points from the document.\n\n"
            "**Suggested Follow-ups:**\n"
            "• What are the key topics in this document?\n"
            "• Can you provide a summary of the main points?\n"
            "• What is the main purpose of this knowledge base?"
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": greeting_response,
            "cached": False,
            "sources": [],
        })
        st.rerun()

    # 2. Contextualize query with conversational history
    with st.spinner("Processing query..."):
        search_query = contextualize_question(
            question=active_question,
            chat_history=st.session_state.messages[:-1],
            llm=llm,
        )

        # 3. Check Semantic Cache (FAISS + JSON)
        semantic_hit = check_semantic_cache(
            question=search_query,
            doc_hash=st.session_state.document_hash,
            threshold=SEMANTIC_SIMILARITY_THRESHOLD,
        )

    if semantic_hit:
        st.session_state.messages.append({
            "role": "assistant",
            "content": semantic_hit["response"],
            "cached": True,
            "score": semantic_hit.get("similarity_score", 1.0),
            "sources": semantic_hit.get("sources", []),
        })
        st.rerun()

    # 4. Cache Miss -> Route Intent: Global Document Summary vs Specific Topic Search
    else:
        with st.spinner("Analyzing document..."):
            if is_global_summary_query(search_query):
                retrieved_docs = get_document_outline_context(
                    st.session_state.document_chunks,
                    max_chunks=5,
                )
            else:
                candidate_docs = retriever.invoke(search_query)
                retrieved_docs = rerank_documents_with_guardrail(
                    query=search_query,
                    docs=candidate_docs,
                    top_k=3,
                    threshold=GUARDRAIL_CONFIDENCE_THRESHOLD,
                )

        # Guardrail triggered: candidates scored below logit threshold
        if not retrieved_docs:
            fallback = (
                "I could not find any information relevant to your inquiry in the uploaded document. "
                "Please try rephrasing your question or check if the topic is covered in the document."
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": fallback,
                "cached": False,
                "sources": [],
            })
            st.rerun()

        # # Context passes guardrail -> Stream LLM response
        # context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        # messages = prompt.format_messages(context=context, input=search_query)

        # st.markdown(
        #     """
        #     <div class="chat-assistant-container">
        #         <div class="assistant-header-row">
        #             <span class="assistant-label">✦ Document AI</span>
        #         </div>
        #     </div>
        #     """,
        #     unsafe_allow_html=True,
        # )
        # response_placeholder = st.empty()
        # full_response = ""

        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        messages = prompt.format_messages(context=context, input=search_query)

        response_placeholder = st.empty()
        full_response = ""

        for chunk in llm.stream(messages):
            if chunk.content:
                full_response += chunk.content
                response_placeholder.markdown(full_response)

        for chunk in llm.stream(messages):
            if chunk.content:
                full_response += chunk.content
                response_placeholder.markdown(full_response)

        serializable_sources = [
            {
                "source": d.metadata.get("source", "Unknown"),
                "page": d.metadata.get("page_number", d.metadata.get("page", "N/A")),
            }
            for d in retrieved_docs
        ]

        # Store completed generation in semantic cache
        store_semantic_cache(
            question=search_query,
            response=full_response,
            sources=serializable_sources,
            doc_hash=st.session_state.document_hash,
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "cached": False,
            "sources": serializable_sources,
        })
        st.rerun()

# Anchor for smooth auto-scroll
st.markdown('<div id="chat-bottom-anchor"></div>', unsafe_allow_html=True)
if len(st.session_state.messages) > 1:
    inject_scroll_button()