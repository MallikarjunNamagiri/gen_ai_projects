import hashlib
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from src.config import FAISS_INDEX_DIR, DATA_DIR
from src.core.models import get_embeddings

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

def save_and_build_vectorstore(uploaded_file):
    saved_file_path = DATA_DIR / uploaded_file.name
    with open(saved_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return load_document_into_vectorstore(saved_file_path)