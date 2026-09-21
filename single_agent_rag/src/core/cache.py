import json
import hashlib
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.config import QUERY_CACHE_DIR, QUERY_JSON_PATH, SEMANTIC_SIMILARITY_THRESHOLD
from src.core.models import get_embeddings

def load_query_json() -> dict:
    if QUERY_JSON_PATH.exists():
        try:
            with open(QUERY_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_query_json(data: dict):
    with open(QUERY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_query_cache_vectorstore():
    embeddings = get_embeddings()
    index_file = QUERY_CACHE_DIR / "index.faiss"
    pickle_file = QUERY_CACHE_DIR / "index.pkl"
    if index_file.exists() and pickle_file.exists():
        try:
            return FAISS.load_local(
                str(QUERY_CACHE_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            return None
    return None

def check_semantic_cache(question: str, doc_hash: str, threshold: float = SEMANTIC_SIMILARITY_THRESHOLD):
    query_vs = get_query_cache_vectorstore()
    if query_vs is None:
        return None

    try:
        results = query_vs.similarity_search_with_relevance_scores(question, k=1)
    except Exception:
        return None

    if not results:
        return None

    matched_doc, similarity_score = results[0]
    if similarity_score >= threshold:
        if matched_doc.metadata.get("doc_hash") == doc_hash:
            query_id = matched_doc.metadata.get("query_id")
            cache_json = load_query_json()
            if query_id in cache_json:
                entry = cache_json[query_id]
                entry["similarity_score"] = similarity_score
                return entry

    return None

def store_semantic_cache(question: str, response: str, sources: list, doc_hash: str):
    embeddings = get_embeddings()
    query_id = hashlib.sha256(f"{doc_hash}_{question}".encode("utf-8")).hexdigest()

    cache_json = load_query_json()
    cache_json[query_id] = {
        "original_question": question,
        "response": response,
        "sources": sources,
        "doc_hash": doc_hash,
    }
    save_query_json(cache_json)

    query_doc = Document(
        page_content=question,
        metadata={"query_id": query_id, "doc_hash": doc_hash},
    )

    query_vs = get_query_cache_vectorstore()
    if query_vs is None:
        query_vs = FAISS.from_documents([query_doc], embeddings)
    else:
        query_vs.add_documents([query_doc])

    query_vs.save_local(str(QUERY_CACHE_DIR))