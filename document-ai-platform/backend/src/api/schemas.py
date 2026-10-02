from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str
    document_hash: str
    chat_history: Optional[List[dict]] = []

class DocumentInfo(BaseModel):
    name: str
    hash: str
    chunk_count: int