from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None # Useful later for tracking persistent agent memory

class ChatResponse(BaseModel):
    response: str
    status: str = "success"