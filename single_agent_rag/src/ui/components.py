import streamlit as st
from src.services.rag import extract_answer_and_followups

def trigger_followup(question_text: str):
    st.session_state.pending_question = question_text

def render_assistant_response(content: str, cached: bool = False, score: float = 1.0, sources: list = None, is_latest: bool = False, msg_idx: int = 0):
    # Ensure there is always a valid right-hand element for the flex container
    if cached:
        score_pct = int(score * 100)
        badge_html = f'<span class="cache-badge">⚡ Semantic Cache ({score_pct}%)</span>'
    else:
        badge_html = '<span style="display:none;"></span>'

    header_html = (
        '<div class="chat-assistant-container">'
        '<div class="assistant-header-row">'
        '<span class="assistant-label">✦ Document AI</span>'
        f'{badge_html}'
        '</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)
    
    main_answer, followups = extract_answer_and_followups(content)
    st.markdown(main_answer)

    if followups and is_latest:
        st.markdown(
            '<div class="followup-wrapper"><div class="followup-header">✦ Suggested Follow-ups</div></div>', 
            unsafe_allow_html=True
        )
        cols = st.columns(len(followups))
        for idx, (col, q_text) in enumerate(zip(cols, followups)):
            with col:
                st.button(
                    f"💬 {q_text}",
                    key=f"followup_btn_{msg_idx}_{idx}",
                    on_click=trigger_followup,
                    args=(q_text,),
                    use_container_width=True,
                )

    if sources:
        unique_sources = {}
        for s in sources:
            src = s.get("source", "Unknown") if isinstance(s, dict) else getattr(s, "metadata", {}).get("source", "Unknown")
            pg = s.get("page", "N/A") if isinstance(s, dict) else getattr(s, "metadata", {}).get("page_number", "N/A")
            unique_sources[(src, pg)] = True

        if unique_sources:
            with st.expander(f"Sources used · {len(unique_sources)}"):
                for source, page in unique_sources:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-name">▣ &nbsp; {source}</div>'
                        f'<div class="source-meta">Page {page}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )