from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000) # Increased max_length
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    
class HealthResponse(BaseModel):
    status: str
    version: str