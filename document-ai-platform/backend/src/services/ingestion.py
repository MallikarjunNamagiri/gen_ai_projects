import hashlib
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from backend.src.config import FAISS_INDEX_DIR, DATA_DIR
from src.core.models import get_embeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
import os

from src.config import FAISS_INDEX_DIR, DATA_DIR

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def calculate_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def load_document_into_vectorstore(file_path: Path):
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    file_hash = calculate_file_hash(file_bytes)

    file_extension = file_path.suffix.lower()
    if file_extension == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif file_extension == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = file_path.name
        chunk.metadata["chunk_index"] = i
        if "page" in chunk.metadata:
            chunk.metadata["page_number"] = chunk.metadata["page"] + 1

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(FAISS_INDEX_DIR))

    return vectorstore, chunks, file_path.name, file_hash

def chunk_document(file_obj):
    file_name = file_obj.name
    file_path = Path(DATA_DIR) / file_name
    file_extension = file_path.suffix.lower()
    if file_extension == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif file_extension == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = file_path.name
        chunk.metadata["chunk_index"] = i
        if "page" in chunk.metadata:
            chunk.metadata["page_number"] = chunk.metadata["page"] + 1

    return chunks

def save_and_build_vectorstore(file_obj, chunks=None, embeddings=None):
    file_bytes = file_obj.getvalue() if hasattr(file_obj, "getvalue") else file_obj.read()
    file_hash = calculate_file_hash(file_bytes)
    file_name = file_obj.name

    if embeddings is None:
        from src.core.models import get_embeddings
        embeddings = get_embeddings()

    if chunks is None:
        # Load and chunk the document directly from file_obj
        chunks = chunk_document(file_obj)
    # Collection partitioned or tagged by document hash
    qdrant_vs = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name="enterprise_documents",
        prefer_grpc=False
    )
    return qdrant_vs, chunks, file_name, file_hash