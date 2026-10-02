from functools import lru_cache
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from src.config import GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL, RERANKER_MODEL

@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(RERANKER_MODEL)

@lru_cache(maxsize=1)
def get_llm():
    if not GROQ_API_KEY:
        return None
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,
        streaming=True,
    )