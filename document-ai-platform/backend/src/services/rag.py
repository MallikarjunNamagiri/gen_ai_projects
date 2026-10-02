import re
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from src.core.models import get_llm

def create_rag_chain(vectorstore, chunks):
    llm = get_llm()
    if llm is None:
        return None

    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 30,
            "lambda_mult": 0.7,
        },
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    prompt = ChatPromptTemplate.from_template(
    """You are a helpful and knowledgeable AI document assistant.
        Your answers must strictly rely on the provided context below.

        Guidelines:
        1. Provide an accurate, well-structured, and clear answer.
        2. If the user greets you or includes conversational pleasantries, respond warmly before addressing their inquiry.
        3. If the user asks about the document's purpose, key topics, or scope, summarize the overarching themes present in the context.
        4. If the question cannot be answered from the context, state that the information is not covered in the document.

        After your answer, provide 2 or 3 short, relevant follow-up questions that the user might want to ask next based on the document context, formatted under the header '**Suggested Follow-ups:**'.

        <context>
        {context}
        </context>

        Question:
        {input}
        """
        )

    return prompt, hybrid_retriever, llm

def extract_answer_and_followups(full_text: str):
    pattern = r"(?i)\*{0,2}suggested follow-?ups?:?\*{0,2}"
    parts = re.split(pattern, full_text, maxsplit=1)
    
    if len(parts) == 1:
        return full_text.strip(), []
    
    main_answer = parts[0].strip()
    followup_section = parts[1].strip()
    
    questions = []
    for line in followup_section.split("\n"):
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^[\s*•\-\d.]+", "", line).strip()
        cleaned = cleaned.replace("**", "").replace("*", "").strip(' "\'')
        if cleaned:
            if not cleaned.endswith("?"):
                cleaned += "?"
            questions.append(cleaned)
    
    return main_answer, questions[:3]