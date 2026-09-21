import streamlit as st
import streamlit.components.v1 as components

def inject_custom_css():
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background: radial-gradient(circle at 70% 5%, rgba(230, 177, 57, 0.08), transparent 25%),
                        radial-gradient(circle at 95% 60%, rgba(230, 177, 57, 0.035), transparent 30%),
                        #080a0e !important;
            color: #f5f5f5;
        }
        [data-testid="stAppViewContainer"] { padding-top: 0 !important; }
        [data-testid="stHeader"] { background: transparent !important; }
        .main .block-container {
            max-width: 1100px;
            padding-top: 36px;
            padding-bottom: 170px !important;
            padding-left: 48px;
            padding-right: 48px;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b0e13 0%, #090c11 100%) !important;
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        section[data-testid="stSidebar"] > div { padding: 22px 18px 25px 18px; }
        .brand {
            display: flex; align-items: center; gap: 12px;
            padding-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.07);
        }
        .brand-icon {
            width: 46px; height: 46px; display: flex; align-items: center; justify-content: center;
            border-radius: 12px; background: linear-gradient(135deg, #f5c34d, #d99b22);
            color: #121212; font-size: 24px; box-shadow: 0 10px 30px rgba(232,185,79,0.16);
        }
        .brand-title { font-size: 19px; font-weight: 800; color: #ffffff; letter-spacing: -0.4px; }
        .brand-subtitle { margin-top: 2px; font-size: 11px; color: #7d8491; }
        .side-section-title {
            display: flex; align-items: center; gap: 8px; margin: 20px 0 10px 0;
            color: #8f96a3; font-size: 11px; font-weight: 750; text-transform: uppercase; letter-spacing: 1.1px;
        }
        .pipeline-card {
            background: rgba(18,22,29,0.92); border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px; padding: 12px 14px; display: flex; flex-direction: column; gap: 10px;
        }
        .pipeline-item { display: flex; align-items: center; justify-content: space-between; font-size: 12px; }
        .pipeline-label { color: #7f8694; }
        .pipeline-value { color: #f0f1f3; font-weight: 600; display: flex; align-items: center; gap: 6px; }
        .pill { display: inline-flex; align-items: center; gap: 5px; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; }
        .pill-green { background: rgba(85, 223, 145, 0.12); color: #55df91; }
        .pill-amber { background: rgba(240, 192, 79, 0.12); color: #f0c04f; }
        .pill-gray { background: rgba(255, 255, 255, 0.08); color: #cbd1db; }
        .active-file-box {
            background: rgba(231, 182, 62, 0.08); border: 1px solid rgba(231, 182, 62, 0.25);
            border-radius: 9px; padding: 10px 12px; margin-top: 8px;
        }
        .active-file-name { color: #f0f1f3; font-size: 12px; font-weight: 600; word-break: break-all; }
        .active-file-status { margin-top: 4px; color: #55df91; font-size: 11px; display: flex; align-items: center; gap: 5px; }
        .hero { position: relative; padding: 8px 0 24px 0; }
        .eyebrow { color: #e7b63e; font-size: 11px; font-weight: 850; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px; }
        .hero-title { margin: 0; color: #ffffff; font-size: clamp(30px, 4vw, 44px); font-weight: 850; line-height: 1.1; letter-spacing: -1.6px; }
        .hero-title .gold { color: #e9b83f; }
        .chat-user { display: flex; justify-content: flex-end; margin: 16px 0 8px; }
        .chat-user-bubble {
            max-width: 75%; padding: 12px 16px; background: #1a1f28;
            border: 1px solid rgba(255,255,255,0.08); border-radius: 15px 15px 4px 15px;
            color: #eef0f3; font-size: 14px; line-height: 1.6;
        }
        .chat-assistant-container {
            display: flex; flex-direction: column; margin: 8px 0 16px; background: #11151b;
            border: 1px solid rgba(255,255,255,0.07); border-radius: 4px 15px 15px 15px;
            padding: 16px 20px; max-width: 90%;
        }
        .assistant-header-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
        .assistant-label { color: #e7b63e; font-size: 10px; font-weight: 850; text-transform: uppercase; letter-spacing: 1.3px; }
        .cache-badge {
            font-size: 10px; color: #55df91; background: rgba(85, 223, 145, 0.12);
            border: 1px solid rgba(85, 223, 145, 0.25); padding: 2px 7px; border-radius: 5px; font-weight: 600;
        }
        .followup-wrapper { margin-top: 14px; margin-bottom: 6px; }
        .followup-header {
            color: #e7b63e; font-size: 11px; font-weight: 750; text-transform: uppercase;
            letter-spacing: 1.1px; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;
        }
        div[data-testid="stHorizontalBlock"] button {
            background: rgba(18, 22, 29, 0.9) !important; border: 1px solid rgba(231, 182, 62, 0.35) !important;
            color: #dfe2e7 !important; border-radius: 12px !important; padding: 6px 14px !important;
            font-size: 12px !important; text-align: left !important; line-height: 1.4 !important;
        }
        .source-card { padding: 9px 12px; margin: 4px 0; background: #0c1015; border: 1px solid rgba(255,255,255,0.07); border-radius: 8px; }
        .source-name { color: #dce0e6; font-size: 12px; font-weight: 600; }
        .source-meta { margin-top: 2px; color: #707885; font-size: 10px; }
        .empty-state { text-align: center; padding: 60px 20px; }
        .empty-icon {
            display: inline-flex; align-items: center; justify-content: center; width: 58px; height: 58px;
            border-radius: 16px; background: rgba(231,182,62,0.10); border: 1px solid rgba(231,182,62,0.17);
            color: #e8b63f; font-size: 24px;
        }
        .empty-title { margin-top: 14px; color: #ffffff; font-size: 20px; font-weight: 750; }
        .empty-description { max-width: 460px; margin: 8px auto 0; color: #777f8d; font-size: 13px; line-height: 1.6; }
        div[data-testid="stBottom"] {
            background: linear-gradient(180deg, rgba(8, 10, 14, 0) 0%, rgba(8, 10, 14, 0.92) 30%, #080a0e 100%) !important;
            padding-top: 28px !important; padding-bottom: 20px !important; backdrop-filter: blur(8px) !important;
        }
        div[data-testid="stChatInput"] {
            background: #ffffff !important; border-radius: 12px !important; padding: 4px 6px !important;
        }
        div[data-testid="stChatInput"] textarea { color: #111418 !important; }
        #scrollToBottomBtn {
            position: fixed; bottom: 92px; right: 40px; z-index: 99999; background: #19202a;
            color: #f5c34d; border: 1px solid rgba(245, 195, 77, 0.4); border-radius: 50%;
            width: 38px; height: 38px; cursor: pointer; display: flex; align-items: center; justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def inject_scroll_button():
    components.html(
        """
        <button id="scrollToBottomBtn" title="Scroll to latest response">↓</button>
        <script>
            const doc = window.parent.document;
            const btn = doc.getElementById("scrollToBottomBtn");
            if (btn) {
                btn.onclick = function() {
                    const anchor = doc.getElementById("chat-bottom-anchor");
                    if (anchor) {
                        anchor.scrollIntoView({ behavior: 'smooth', block: 'end' });
                    }
                };
            }
        </script>
        """,
        height=0,
        width=0,
    )