import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends

from app.schemas.embedding import EmbeddingProcessResponse, ChunkEmbedding
from app.agents.embedding_agent import EmbeddingAgent
from app.tools.document_tools import extract_text_from_bytes
from app.core.logging import get_logger
from app.core.users import current_active_user
from app.models.user import User

router = APIRouter()
logger = get_logger(__name__)

def get_embedding_agent() -> EmbeddingAgent:
    return EmbeddingAgent()

@router.post(
    "/process",
    response_model=EmbeddingProcessResponse,
    summary="Upload a document and get chunk embeddings",
)
async def process_embeddings(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    agent: EmbeddingAgent = Depends(get_embedding_agent),
    current_user: User = Depends(current_active_user),
):
    logger.info("route_embedding_process", filename=file.filename)
    
    try:
        file_bytes = await file.read()
        extracted_text = extract_text_from_bytes(file_bytes, file.filename)
        
        result = await agent.run(
            raw_content=extracted_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunk_embeddings = [
            ChunkEmbedding(chunk_text=chunk, embedding=embedding)
            for chunk, embedding in zip(result["chunks"], result["embeddings"])
        ]
        
        return EmbeddingProcessResponse(
            document_id=result["document_id"],
            total_chunks=len(chunk_embeddings),
            data=chunk_embeddings,
            status=result["status"]
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("route_embedding_process_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
