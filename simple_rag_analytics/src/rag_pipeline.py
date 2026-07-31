"""
Simplest RAG pipeline for generic analytics documents.
Uses local embeddings (sentence-transformers) + Chroma.
No paid API required for retrieval.
"""

from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------- 1. Load documents ----------
DATA_DIR = Path(__file__).parent.parent / "data"
loader = DirectoryLoader(str(DATA_DIR), glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()
print(f"Loaded {len(docs)} documents")

# ---------- 2. Chunk ----------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# ---------- 3. Embed + store ----------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------- 4. Simple RAG chain (retrieval only + print context) ----------
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# For full generation you can later plug any LLM.
# For the 70-min version we just show retrieved context + a template answer.
rag_prompt = ChatPromptTemplate.from_template(
    """You are an analytics assistant. Use only the following context to answer the question.
If the answer is not in the context, say "I don't have enough information in the documents."

Context:
{context}

Question: {question}

Answer:"""
)

# Retrieval-only demo (no LLM call needed for first commit)
def ask(question: str):
    docs = retriever.invoke(question)
    context = format_docs(docs)
    print("=" * 60)
    print(f"QUESTION: {question}")
    print("-" * 60)
    print("RETRIEVED CONTEXT:")
    print(context)
    print("=" * 60)
    return context

if __name__ == "__main__":
    # Example queries – replace with your own
    ask("What were the main drivers of customer churn?")
    ask("Which product features have the highest adoption?")
    ask("What is the current inventory turnover situation?")