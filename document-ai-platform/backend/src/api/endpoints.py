import json
import asyncio
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import QueryRequest
from src.core.models import get_llm
from src.core.cache import check_semantic_cache, store_semantic_cache, get_cached_queries_count
from src.services.router import is_greeting_query, contextualize_question
from src.services.evaluator import run_ragas_evaluation
from src.services.ingestion import process_pdf_in_memory, index_chunks_to_qdrant
from src.services.qdrant_ops import get_registered_documents_from_qdrant, search_qdrant_with_doc_filter
from src.config import GROQ_MODEL, GUARDRAIL_CONFIDENCE_THRESHOLD, SEMANTIC_SIMILARITY_THRESHOLD
from langchain_core.prompts import ChatPromptTemplate

router = APIRouter()

class SelectDocRequest(BaseModel):
    doc_hash: str
    doc_name: str

class EvalRequest(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = ""

# 1. Zero-Disk Document List
@router.get("/documents")
async def list_documents():
    docs = get_registered_documents_from_qdrant()
    return {"files": [d["name"] for d in docs], "documents": docs}

# 2. In-Memory Upload -> Qdrant Metadata Layer
@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")

    file_bytes = await file.read()
    chunks, doc_hash = process_pdf_in_memory(file_bytes, file.filename)
    indexed_count = index_chunks_to_qdrant(chunks, doc_hash, file.filename)

    return {
        "name": file.filename,
        "hash": doc_hash,
        "chunks": indexed_count
    }

# 3. Stateless Streaming with Qdrant Metadata Retrieval
@router.post("/chat/stream")
async def chat_stream(payload: QueryRequest):
    question = payload.question.strip()
    doc_hash = payload.document_hash

    # Greeting Interceptor
    if is_greeting_query(question):
        async def greeting_generator():
            yield f"data: {json.dumps({'type': 'token', 'content': 'Hello! Ready to analyze your document. What would you like to know?'})}\n\n"
            yield f"data: {json.dumps({'type': 'meta', 'cached': False, 'sources': []})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(greeting_generator(), media_type="text/event-stream")

    if not doc_hash:
        raise HTTPException(status_code=400, detail="No active document selected. Please select or upload a document.")

    llm = get_llm()
    search_query = contextualize_question(question, payload.chat_history, llm)

    # Check Upstash Semantic Cache
    cache_hit = check_semantic_cache(search_query, doc_hash, SEMANTIC_SIMILARITY_THRESHOLD)
    if cache_hit:
        async def cache_generator():
            yield f"data: {json.dumps({'type': 'token', 'content': cache_hit['response']})}\n\n"
            yield f"data: {json.dumps({'type': 'meta', 'cached': True, 'score': cache_hit.get('similarity_score', 1.0), 'sources': cache_hit.get('sources', [])})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(cache_generator(), media_type="text/event-stream")

    # Fetch relevant chunks directly from Qdrant metadata
    docs = search_qdrant_with_doc_filter(search_query, doc_hash, top_k=4)

    if not docs:
        async def empty_generator():
            fallback = "I could not find relevant context in the selected document."
            yield f"data: {json.dumps({'type': 'token', 'content': fallback})}\n\n"
            yield f"data: {json.dumps({'type': 'meta', 'cached': False, 'sources': []})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty_generator(), media_type="text/event-stream")

    prompt_tmpl = ChatPromptTemplate.from_messages([
        ("system", "You are an enterprise document assistant. Answer the question using ONLY the provided document excerpts. If not present, state so clearly.\n\nContext:\n{context}"),
        ("human", "{input}")
    ])

    context_str = "\n\n".join(d.page_content for d in docs)
    messages = prompt_tmpl.format_messages(context=context_str, input=search_query)

    async def event_generator():
        full_text = ""
        for chunk in llm.stream(messages):
            if chunk.content:
                full_text += chunk.content
                yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
                await asyncio.sleep(0.005)

        sources = [{"source": d.metadata.get("source"), "page": d.metadata.get("page_number", 1)} for d in docs]
        store_semantic_cache(search_query, full_text, sources, doc_hash)

        context_texts = [d.page_content for d in docs]
        yield f"data: {json.dumps({'type': 'meta', 'cached': False, 'sources': sources, 'contexts': context_texts})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/status")
async def get_system_status():
    cached_count = get_cached_queries_count()
    return {
        "llm_provider": "Groq",
        "llm_connected": True,
        "model": GROQ_MODEL,
        "vector_db": "Qdrant Cloud (Payload Layer)",
        "vector_status": "Ready",
        "retrieval": "Filtered Vector + Metadata Payload",
        "guardrail": f"Cutoff ({GUARDRAIL_CONFIDENCE_THRESHOLD})",
        "semantic_cache_count": f"{cached_count} cached (Upstash)",
        "match_threshold": f"{int(SEMANTIC_SIMILARITY_THRESHOLD * 100)}%"
    }

@router.post("/evaluate")
async def evaluate_rag_turn(payload: EvalRequest):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, run_ragas_evaluation, payload.question, payload.answer, payload.contexts, payload.ground_truth
    )