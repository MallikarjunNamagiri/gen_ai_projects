import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

GLOBAL_PATTERNS = [
    # Document outline & summary patterns
    r"\bkey topics\b",
    r"\btable of contents\b",
    r"\bwhat topics\b",
    r"\bsummarize\b",
    r"\bsummary\b",
    r"\boverview\b",
    r"\bmain points\b",
    r"\bwhat is this document about\b",
    r"\bwhat does this document cover\b",
    r"\bwhat is included in this document\b",

    # Meta-queries about the bot, scope, and capabilities
    r"\bknowledge base\b",
    r"\bwhat can you help (me )?with\b",
    r"\bwhat are you trained (for|to|on)\b",
    r"\bpurpose of (your|this)\b",
    r"\bscope of (your|this)\b",
    r"\bwhat tasks (can|do) you\b",
    r"\bwhat do you know\b",
    r"\bwhat can you do\b",
]

GREETING_PATTERNS = [
    r"^hi\b",
    r"^hello\b",
    r"^hey\b",
    r"^good\s+(morning|afternoon|evening)\b",
    r"^greetings\b",
    r"^howdy\b",
]


REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Given a chat history and the latest user question which might reference context in the chat history, 
            formulate a standalone question which can be understood without the chat history. 
            Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])



def is_greeting_query(query: str) -> bool:
    q_clean = query.lower().strip()
    # Remove punctuation before testing
    q_clean = re.sub(r"[^\w\s]", "", q_clean)
    return any(re.search(pat, q_clean) for pat in GREETING_PATTERNS)


def is_global_summary_query(query: str) -> bool:
    q_lower = query.lower().strip()
    return any(re.search(pat, q_lower) for pat in GLOBAL_PATTERNS)

def get_document_outline_context(chunks: list, max_chunks: int = 5) -> list:
    total = len(chunks)
    if total <= max_chunks:
        return chunks

    selected_indices = [0, 1]
    mid1 = total // 3
    mid2 = (2 * total) // 3
    selected_indices.extend([mid1, mid2, total - 1])

    seen = set()
    sample_chunks = []
    for idx in selected_indices:
        if idx not in seen and idx < total:
            seen.add(idx)
            sample_chunks.append(chunks[idx])
    return sample_chunks

def contextualize_question(question: str, chat_history: list, llm) -> str:
    if not chat_history or llm is None:
        return question

    history_messages = []
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            history_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_messages.append(AIMessage(content=msg["content"]))

    if not history_messages:
        return question

    try:
        rewrite_chain = REWRITE_PROMPT | llm
        response = rewrite_chain.invoke({
            "chat_history": history_messages,
            "input": question
        })
        standalone_query = response.content.strip()
        return standalone_query if standalone_query else question
    except Exception:
        return question