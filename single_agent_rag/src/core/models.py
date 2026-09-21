import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from src.config import GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL, RERANKER_MODEL

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource(show_spinner=False)
def get_reranker():
    return CrossEncoder(RERANKER_MODEL)

@st.cache_resource(show_spinner=False)
def get_llm():
    if not GROQ_API_KEY:
        return None
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,
        streaming=True,
    )