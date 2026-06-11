from pydantic import BaseModel
from typing import List

class ChunkEmbedding(BaseModel):
    chunk_text: str
    embedding: List[float]

class EmbeddingProcessResponse(BaseModel):
    document_id: str
    total_chunks: int
    data: List[ChunkEmbedding]
    status: str
