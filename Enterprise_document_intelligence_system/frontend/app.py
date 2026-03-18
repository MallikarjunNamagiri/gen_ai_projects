# ============================================================
# frontend/app.py
# Streamlit UI for the CLT RAG pipeline.
# Run with:  streamlit run frontend/app.py
# Requires the FastAPI backend running on port 8000.
# ============================================================

import streamlit as st
import requests
import json
from datetime import datetime

API_BASE = "http://127.0.0.1:8000"

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="CLT RAG Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px;
        margin: 6px 0;
        border-left: 4px solid #7c3aed;
    }
    .answer-box {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #10b981;
        margin: 12px 0;
        font-size: 15px;
        line-height: 1.7;
    }
    .cache-hit {
        background: #1e1e2e;
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 10px 16px;
        font-size: 13px;
        color: #f59e0b;
        margin-bottom: 8px;
    }
    .source-tag {
        background: #2d2d44;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        color: #a0aec0;
        display: inline-block;
        margin: 3px 2px;
    }
    .pass-badge  { color: #10b981; font-weight: bold; }
    .fail-badge  { color: #ef4444; font-weight: bold; }
    .stTextArea textarea { font-size: 15px; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────

def get_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def get_cache_stats():
    try:
        r = requests.get(f"{API_BASE}/cache/stats", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def get_index_stats():
    try:
        r = requests.get(f"{API_BASE}/index/stats", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def clear_cache():
    try:
        r = requests.delete(f"{API_BASE}/cache", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def run_query(query: str, top_k: int, threshold: float):
    payload = {"query": query, "top_k": top_k, "retrieval_threshold": threshold}
    r = requests.post(f"{API_BASE}/query", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


# ── Sidebar ───────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/64/combo-chart.png", width=52)
    st.title("CLT RAG Assistant")
    st.caption("Customer Lifetime Analytics · Local LLM")
    st.divider()

    # Backend health
    st.subheader("🔌 Backend Status")
    health = get_health()
    if health:
        status_color = "🟢" if health.get("status") == "ok" else "🟡"
        st.markdown(f"{status_color} **API**: Connected")
        st.markdown(
            f"{'🟢' if health.get('ollama_ready') else '🔴'} "
            f"**Ollama** (`{health.get('ollama_model', '—')}`)"
        )
        st.markdown(
            f"{'🟢' if health.get('index_ready') else '🔴'} **FAISS Index**"
        )
        st.markdown(
            f"{'🟢' if health.get('cache_ready') else '🔴'} **Semantic Cache**"
        )
    else:
        st.error("❌ Cannot reach backend.\nStart with:\n```\nuvicorn backend.main:app --reload\n```")

    st.divider()

    # Query settings
    st.subheader("⚙️ Query Settings")
    top_k = st.slider(
        "Top-K chunks retrieved", min_value=1, max_value=10,
        value=5, help="Number of context chunks passed to the LLM."
    )
    threshold = st.slider(
        "Retrieval threshold", min_value=0.1, max_value=0.9,
        value=0.3, step=0.05,
        help="Minimum cosine similarity for a chunk to qualify. Lower = more permissive."
    )

    st.divider()

    # Index stats
    st.subheader("📦 Index")
    idx = get_index_stats()
    if idx:
        st.metric("Vectors indexed", idx.get("total_vectors", "—"))
        st.caption(f"Model: `{idx.get('model_name', '—')}`")

    st.divider()

    # Cache stats + clear button
    st.subheader("⚡ Semantic Cache")
    cs = get_cache_stats()
    if cs:
        col1, col2 = st.columns(2)
        col1.metric("Lookups", cs.get("total_lookups", 0))
        col2.metric("Hits",    cs.get("total_hits", 0))
        hit_rate = cs.get("hit_rate", 0)
        st.progress(
            float(hit_rate),
            text=f"Hit rate: {hit_rate * 100:.1f}%"
        )

    if st.button("🗑️ Clear Cache", use_container_width=True):
        if clear_cache():
            st.success("Cache cleared!")
            st.rerun()
        else:
            st.error("Failed to clear cache.")

    st.divider()
    st.caption("💡 Tip: Rephrase a previous question to see a cache hit.")


# ── Main area ─────────────────────────────────────────────────

st.header("📊 CLT Analytics Q&A")
st.markdown(
    "Ask questions about churn rate, survival rate, active customers, "
    "gross adds, and cohort trends across tenures."
)

# Suggested questions
with st.expander("💡 Example questions", expanded=False):
    examples = [
        "What is the churn rate at tenure 10 for 5G_Plan Online Consumer customers?",
        "How does the survival rate change between tenure 50 and tenure 100?",
        "How many active customers remain at tenure 100 for the Online 5G_Plan cohort?",
        "What was the gross add count at tenure 0?",
        "Which tenure has the highest churn rate?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["prefill"] = ex

# Chat history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Query input
prefill = st.session_state.pop("prefill", "")
query   = st.text_area(
    "Your question",
    value=prefill,
    placeholder="e.g. What is the survival rate at tenure 30?",
    height=90,
    label_visibility="collapsed"
)

col_ask, col_clear = st.columns([5, 1])
with col_ask:
    ask_clicked = st.button("🔍 Ask", type="primary", use_container_width=True, disabled=not health)
with col_clear:
    if st.button("🗂️ Clear history", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

# ── Query execution ───────────────────────────────────────────

if ask_clicked and query.strip():
    with st.spinner("Thinking..."):
        try:
            result = run_query(query.strip(), top_k, threshold)

            # Store in session history
            st.session_state["history"].append({
                "query":  query.strip(),
                "result": result,
                "time":   datetime.now().strftime("%H:%M:%S")
            })

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the backend. Is `uvicorn backend.main:app` running?")
        except Exception as e:
            st.error(f"Error: {e}")

elif ask_clicked and not query.strip():
    st.warning("Please enter a question.")


# ── History display ───────────────────────────────────────────

for item in reversed(st.session_state["history"]):
    q      = item["query"]
    result = item["result"]
    ts     = item["time"]

    st.markdown(f"---")
    st.markdown(f"**🕐 {ts} — {q}**")

    # Cache hit badge
    if result.get("served_from_cache"):
        sim = result.get("cache_similarity", 0)
        oq  = result.get("original_query", "")
        st.markdown(
            f'<div class="cache-hit">⚡ Served from cache '
            f'(similarity: {sim:.4f})<br>'
            f'<small>Matched: "{oq[:80]}..."</small></div>',
            unsafe_allow_html=True
        )

    # Answer
    st.markdown(
        f'<div class="answer-box">{result.get("answer", "—")}</div>',
        unsafe_allow_html=True
    )

    # Sources
    sources = result.get("sources", [])
    if sources:
        st.markdown("**Sources:**")
        source_html = "".join(
            f'<span class="source-tag">📁 {s.split("|")[-1].strip()}</span>'
            for s in sources
        )
        st.markdown(source_html, unsafe_allow_html=True)

    # Metrics (only for non-cached responses)
    if not result.get("served_from_cache") and result.get("evaluation"):
        ev = result["evaluation"]
        with st.expander("📈 Evaluation Metrics", expanded=False):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Faithfulness",    f"{ev['faithfulness_score']:.3f}")
            m2.metric("Relevancy",       f"{ev['answer_relevancy_score']:.3f}")
            m3.metric("Retrieval Score", f"{ev['top_retrieval_score']:.3f}")
            m4.metric("Precision",       f"{ev['retrieval_precision']:.3f}")

            passed = ev.get("passed", False)
            badge  = "pass-badge" if passed else "fail-badge"
            label  = "✅ PASSED" if passed else "❌ FAILED"
            st.markdown(
                f'Evaluation: <span class="{badge}">{label}</span>',
                unsafe_allow_html=True
            )

        tok_in  = result.get("input_tokens", 0)
        tok_out = result.get("output_tokens", 0)
        st.caption(
            f"🔢 Tokens — input: {tok_in} | output: {tok_out} | "
            f"retrieval score: {result.get('top_retrieval_score', 0):.4f}"
        )

# Footer
st.divider()
st.caption("CLT RAG · Local Ollama · FAISS · Qdrant · all-MiniLM-L6-v2")
