from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class DocumentProcessRequest(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="Raw text content of the document")
    query: str = Field(..., description="User query about the document")
    thread_id: Optional[str] = Field(default=None, description="Optional session ID")

class DocumentProcessResponse(BaseModel):
    response: str = Field(..., description="The agent's response to the query")
    document_id: str = Field(..., description="The document identifier")
    extracted_data: Dict[str, Any] = Field(default_factory=dict, description="Any structured data extracted during processing")
    thread_id: str = Field(..., description="The session ID")
    total_messages: int = Field(..., description="Total messages in the graph state")

class DocumentChunkRequest(BaseModel):
    content: str = Field(..., description="Raw text content of the document to chunk")
    chunk_size: int = Field(default=1000, description="Maximum number of characters per chunk")
    chunk_overlap: int = Field(default=200, description="Number of characters to overlap between chunks")

class DocumentChunkResponse(BaseModel):
    chunks: list[str] = Field(..., description="List of text chunks")
    total_chunks: int = Field(..., description="Total number of chunks generated")

class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_path: str
    size: int
    created_at: datetime
    status: str
    training_progress: int
    
    model_config = {
        "from_attributes": True
    }
