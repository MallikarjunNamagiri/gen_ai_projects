import io
import hashlib
from typing import List, Tuple
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.core.models import get_embeddings
import os

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "enterprise_documents"

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def ensure_collection_exists(client: QdrantClient, vector_size: int = 384):
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

def process_pdf_in_memory(file_bytes: bytes, filename: str) -> Tuple[List[Document], str]:
    """Extracts text and chunks directly from memory without saving to disk."""
    doc_hash = hashlib.sha256(file_bytes).hexdigest()
    reader = PdfReader(io.BytesIO(file_bytes))
    
    documents = []
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "doc_name": filename,
                        "doc_hash": doc_hash,
                        "page_number": page_idx + 1
                    }
                )
            )

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    
    # Store token estimate in metadata
    for chunk in chunks:
        chunk.metadata["token_count"] = len(chunk.page_content.split())

    return chunks, doc_hash

def index_chunks_to_qdrant(chunks: List[Document], doc_hash: str, filename: str) -> int:
    """Stores vector embeddings along with chunk text and metadata directly inside Qdrant."""
    if not chunks:
        return 0

    client = get_qdrant_client()
    embeddings_model = get_embeddings()
    
    texts = [c.page_content for c in chunks]
    vectors = embeddings_model.embed_documents(texts)
    
    ensure_collection_exists(client, vector_size=len(vectors[0]))
    
    points = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        # Deterministic UUID per chunk
        point_id = hashlib.md5(f"{doc_hash}_{idx}".encode()).hexdigest()
        payload = {
            "page_content": chunk.page_content,
            "doc_name": filename,
            "doc_hash": doc_hash,
            "page_number": chunk.metadata.get("page_number", 1),
            "token_count": chunk.metadata.get("token_count", 0),
            "chunk_index": idx
        }
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i:i + batch_size])

    return len(chunks)