import json
import asyncio
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from langchain_community.vectorstores import FAISS

from src.api.schemas import QueryRequest
from src.services.ingestion import save_and_build_vectorstore
from src.core.models import get_embeddings, get_llm
from src.core.cache import check_semantic_cache, store_semantic_cache
from src.services.router import (
    is_greeting_query, 
    contextualize_question, 
    is_global_summary_query, 
    get_document_outline_context
)
from src.core.guardrails import rerank_documents_with_guardrail
from src.services.rag import create_rag_chain
from src.services.evaluator import run_ragas_evaluation
from src.config import (
    DATA_DIR, 
    GUARDRAIL_CONFIDENCE_THRESHOLD, 
    SEMANTIC_SIMILARITY_THRESHOLD, 
    FAISS_INDEX_DIR,
    GROQ_MODEL, 
    QUERY_JSON_PATH
)
from src.core.cache import get_cached_queries_count

router = APIRouter()

DOC_STATE = {
    "vectorstore": None, 
    "chunks": [], 
    "doc_name": None, 
    "doc_hash": None, 
    "chain": None
}

class SelectDocRequest(BaseModel):
    filename: str

class EvalRequest(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = ""

def init_or_get_chain(vs, chunks):
    """Guarantees a valid (prompt_tmpl, retriever, llm) tuple or creates components directly."""
    chain_obj = create_rag_chain(vs, chunks)
    llm = get_llm()
    if llm is None:
        raise ValueError("GROQ_API_KEY is missing or invalid. Check backend/.env")

    # If create_rag_chain already returned a 3-tuple
    if isinstance(chain_obj, (tuple, list)) and len(chain_obj) == 3:
        return chain_obj[0], chain_obj[1], chain_obj[2]

    # Fallback: Build standard RAG components directly
    from langchain_core.prompts import ChatPromptTemplate
    prompt_tmpl = ChatPromptTemplate.from_messages([
        ("system", "You are an enterprise document assistant. Answer using ONLY the provided context. If unknown, state clearly.\n\nContext:\n{context}"),
        ("human", "{input}")
    ])
    retriever = vs.as_retriever(search_kwargs={"k": 6})
    return prompt_tmpl, retriever, llm

def try_restore_existing_index():
    if DOC_STATE["vectorstore"] is not None and DOC_STATE["chain"] is not None:
        return True
    index_file = Path(FAISS_INDEX_DIR) / "index.faiss"
    if not index_file.exists():
        return False
    try:
        embeddings = get_embeddings()
        vs = FAISS.load_local(
            str(FAISS_INDEX_DIR), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        DOC_STATE["vectorstore"] = vs
        DOC_STATE["chain"] = init_or_get_chain(vs, DOC_STATE.get("chunks", []))
        return True
    except Exception as e:
        print(f"Failed to restore vectorstore from disk: {e}")
        return False

@router.get("/documents")
async def list_documents():
    data_path = Path(DATA_DIR)
    data_path.mkdir(parents=True, exist_ok=True)
    files = [f.name for f in data_path.glob("*.pdf")]
    return {
        "files": files,
        "active_document": DOC_STATE.get("doc_name")
    }

@router.post("/documents/select")
async def select_document(payload: SelectDocRequest):
    file_path = Path(DATA_DIR) / payload.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found in data directory.")
    
    with open(file_path, "rb") as f:
        content = f.read()

    class FileWrapper:
        def __init__(self, name, bytes_data):
            self.name = name
            self._bytes = bytes_data
        def getvalue(self):
            return self._bytes
        def read(self):
            return self._bytes

    file_obj = FileWrapper(payload.filename, content)
    embeddings = get_embeddings()

    # If ingestion.py has a chunking helper, e.g. chunk_document or process_document:
    from src.services.ingestion import chunk_document  # or load_and_chunk
    chunks = chunk_document(file_obj)

    vs, chunks, name, fhash = save_and_build_vectorstore(file_obj, chunks, embeddings)
    DOC_STATE.update({"vectorstore": vs, "chunks": chunks, "doc_name": name, "doc_hash": fhash})
    DOC_STATE["chain"] = init_or_get_chain(vs, chunks)

    return {"name": name, "hash": fhash, "chunks": len(chunks)}

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        data_path = Path(DATA_DIR)
        data_path.mkdir(parents=True, exist_ok=True)
        file_path = data_path / file.filename
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        class FileWrapper:
            def __init__(self, name, bytes_data):
                self.name = name
                self._bytes = bytes_data
            def getvalue(self):
                return self._bytes
            def read(self):
                return self._bytes

        file_obj = FileWrapper(file.filename, content)
        vs, chunks, name, fhash = save_and_build_vectorstore(file_obj)
        DOC_STATE.update({"vectorstore": vs, "chunks": chunks, "doc_name": name, "doc_hash": fhash})
        DOC_STATE["chain"] = init_or_get_chain(vs, chunks)

        return {"name": name, "hash": fhash, "chunks": len(chunks)}
    except Exception as e:
        print(f"Error during upload ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def chat_stream(payload: QueryRequest):
    question = payload.question.strip()

    # 1. Greeting interceptor
    if is_greeting_query(question):
        async def greeting_generator():
            doc_context = f" Ready to analyze **{DOC_STATE['doc_name']}**." if DOC_STATE.get("doc_name") else " Please upload or select a document to get started."
            yield f"data: {json.dumps({'type': 'token', 'content': f'Hello!{doc_context}'})}\n\n"
            yield f"data: {json.dumps({'type': 'meta', 'cached': False, 'sources': []})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(greeting_generator(), media_type="text/event-stream")

    # 2. Check or re-hydrate document & chain
    if (not DOC_STATE.get("vectorstore") or not DOC_STATE.get("chain")) and not try_restore_existing_index():
        raise HTTPException(
            status_code=400, 
            detail="No document active or LLM uninitialized. Please select or upload a document."
        )

    prompt_tmpl, retriever, llm = DOC_STATE["chain"]

    async def event_generator():
        # 3. Contextualize query
        search_query = contextualize_question(question, payload.chat_history, llm)
        
        # 4. Semantic Cache Check
        current_hash = DOC_STATE.get("doc_hash", payload.document_hash)
        cache_hit = check_semantic_cache(search_query, current_hash, SEMANTIC_SIMILARITY_THRESHOLD)

        if cache_hit:
            yield f"data: {json.dumps({'type': 'token', 'content': cache_hit['response']})}\n\n"
            yield f"data: {json.dumps({'type': 'meta', 'cached': True, 'score': cache_hit.get('similarity_score', 1.0), 'sources': cache_hit.get('sources', [])})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # 5. Hybrid Retrieval / Outline
        if is_global_summary_query(search_query) and DOC_STATE.get("chunks"):
            docs = get_document_outline_context(DOC_STATE["chunks"], max_chunks=5)
        else:
            candidates = retriever.invoke(search_query)
            docs = rerank_documents_with_guardrail(search_query, candidates, top_k=3, threshold=GUARDRAIL_CONFIDENCE_THRESHOLD)

        # 6. Guardrail Check
        if not docs:
            fallback = "I could not find any information relevant to your inquiry in the uploaded document."
            yield f"data: {json.dumps({'type': 'token', 'content': fallback})}\n\n"
            yield f"data: {json.dumps({'type': 'meta', 'cached': False, 'sources': []})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # 7. LLM Stream
        context = "\n\n".join(d.page_content for d in docs)
        messages = prompt_tmpl.format_messages(context=context, input=search_query)
        full_text = ""

        try:
            for chunk in llm.stream(messages):
                if chunk.content:
                    full_text += chunk.content
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
                    await asyncio.sleep(0.005)
        except Exception as err:
            error_msg = f"\\n\\n[Error generating response: {err}]"
            yield f"data: {json.dumps({'type': 'token', 'content': error_msg})}\n\n"

        sources = [{"source": d.metadata.get("source", "PDF"), "page": d.metadata.get("page_number", "N/A")} for d in docs]
        if current_hash and full_text:
            store_semantic_cache(search_query, full_text, sources, current_hash)
        
        # Include context chunk texts for frontend Ragas evaluation
        context_texts = [d.page_content for d in docs]
        yield f"data: {json.dumps({'type': 'meta', 'cached': False, 'sources': sources, 'contexts': context_texts})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/status")
async def get_system_status():
    doc_hash = DOC_STATE.get("doc_hash")
    cached_count = get_cached_queries_count(doc_hash)
    total_tokens_saved = cached_count * 450
    estimated_dollars_saved = round((total_tokens_saved / 1_000_000) * 0.79, 4)

    return {
        "llm_provider": "Groq",
        "llm_connected": True,
        "model": GROQ_MODEL,
        "vector_db": "FAISS",
        "vector_status": "Ready" if DOC_STATE.get("vectorstore") else "Waiting File",
        "retrieval": "Hybrid + Router",
        "guardrail": f"Cutoff ({GUARDRAIL_CONFIDENCE_THRESHOLD})",
        "semantic_cache_count": f"{cached_count} cached (Upstash)",
        "estimated_savings": f"${estimated_dollars_saved} saved",
        "match_threshold": f"{int(SEMANTIC_SIMILARITY_THRESHOLD * 100)}%"
    }

@router.post("/evaluate")
async def evaluate_rag_turn(payload: EvalRequest):
    loop = asyncio.get_event_loop()
    scores = await loop.run_in_executor(
        None,
        run_ragas_evaluation,
        payload.question,
        payload.answer,
        payload.contexts,
        payload.ground_truth
    )
    return scores