# api/models.py
from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str


class User(BaseModel):
    id: str
    email: str = None

