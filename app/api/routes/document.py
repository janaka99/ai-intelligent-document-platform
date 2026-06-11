import uuid
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form

from app.schemas.document import DocumentProcessRequest, DocumentProcessResponse, DocumentChunkRequest, DocumentChunkResponse, DocumentResponse
from app.agents.document_agent import DocumentAgent
from app.tools.document_tools import DOCUMENT_TOOLS, chunk_document_text
from app.core.logging import get_logger
from app.core.users import current_active_user
from app.models.user import User
from app.models.document import Document
from app.db.database import get_async_session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import os
from fastapi.responses import StreamingResponse
import json
import asyncio
from datetime import datetime, timezone
from app.models.document import DocumentStatus
from app.db.mongodb import get_mongo_db

router = APIRouter()
logger = get_logger(__name__)

def get_document_agent() -> DocumentAgent:
    """Dependency injection for the document agent."""
    return DocumentAgent(tools=DOCUMENT_TOOLS)

@router.post(
    "/process",
    response_model=DocumentProcessResponse,
    summary="Process a document",
)
async def process_document(
    request: DocumentProcessRequest,
    agent: DocumentAgent = Depends(get_document_agent),
    current_user: User = Depends(current_active_user),
):
    thread_id = request.thread_id or str(uuid.uuid4())

    logger.info(
        "route_document_process",
        thread_id=thread_id,
        document_id=request.document_id,
    )

    try:
        result = await agent._run(
            user_input=request.query,
            thread_id=thread_id,
            document_id=request.document_id,
            raw_content=request.content
        )
        return DocumentProcessResponse(
            response=result["response"],
            document_id=request.document_id,
            extracted_data=result.get("extracted_data", {}),
            thread_id=thread_id,
            total_messages=result["total_messages"],
        )

    except Exception as e:
        logger.exception("route_document_failed", thread_id=thread_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/chunk",
    response_model=DocumentChunkResponse,
    summary="Upload and chunk a document",
)
async def chunk_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    current_user: User = Depends(current_active_user),
):
    logger.info("route_document_chunk", filename=file.filename)
    
    try:
        from app.tools.document_tools import extract_text_from_bytes
        file_bytes = await file.read()
        extracted_text = extract_text_from_bytes(file_bytes, file.filename)
        
        chunks = chunk_document_text(
            text=extracted_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return DocumentChunkResponse(
            chunks=chunks,
            total_chunks=len(chunks)
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("route_document_chunk_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/upload",
    response_model=DocumentResponse,
    summary="Upload a document and save to database",
)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    try:
        # Save file to uploads folder
        file_path = f"uploads/{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
            
        file_size = os.path.getsize(file_path)
        
        # Save to DB
        new_doc = Document(
            filename=file.filename,
            file_path=file_path,
            size=file_size,
            user_id=current_user.id
        )
        session.add(new_doc)
        await session.commit()
        await session.refresh(new_doc)
        
        return new_doc
    except Exception as e:
        logger.exception("route_document_upload_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/",
    response_model=list[DocumentResponse],
    summary="Get all documents for current user",
)
async def get_documents(
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    stmt = select(Document).where(Document.user_id == current_user.id)
    result = await session.execute(stmt)
    documents = result.scalars().all()
    return documents

@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get a single document",
)
async def get_document(
    document_id: str,
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    stmt = select(Document).where(Document.id == document_id, Document.user_id == current_user.id)
    result = await session.execute(stmt)
    document = result.scalars().first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
        
    return document

@router.post(
    "/chunk/{document_id}",
    summary="Chunk an existing document and save to MongoDB with progress",
)
async def chunk_document_by_id(
    document_id: str,
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    stmt = select(Document).where(Document.id == document_id, Document.user_id == current_user.id)
    result = await session.execute(stmt)
    document = result.scalars().first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
        
    if not os.path.exists(document.file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    document.status = DocumentStatus.chunking
    await session.commit()
    
    async def event_generator():
        try:
            from app.tools.document_tools import extract_text_from_bytes
            with open(document.file_path, "rb") as f:
                file_bytes = f.read()
            extracted_text = extract_text_from_bytes(file_bytes, document.filename)
            
            chunks = chunk_document_text(
                text=extracted_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            total_chunks = len(chunks)
            if total_chunks == 0:
                yield f"data: {json.dumps({'progress': 100})}\n\n"
                document.status = DocumentStatus.chunked
                document.training_progress = 100
                await session.commit()
                return

            db = get_mongo_db()
            collection = db.document_chunks
            
            existing_count = await collection.count_documents({"document_id": document_id})
            start_index = existing_count
            
            if start_index >= total_chunks:
                yield f"data: {json.dumps({'progress': 100})}\n\n"
                document.status = DocumentStatus.chunked
                document.training_progress = 100
                await session.commit()
                return

            for i in range(start_index, total_chunks):
                chunk_data = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "content": chunks[i],
                    "created_at": datetime.now(timezone.utc)
                }
                await collection.insert_one(chunk_data)
                
                progress_percentage = int(((i + 1) / total_chunks) * 100)
                document.training_progress = progress_percentage
                await session.commit()
                
                yield f"data: {json.dumps({'progress': progress_percentage})}\n\n"
                await asyncio.sleep(0.01)

            document.status = DocumentStatus.chunked
            await session.commit()
            
        except Exception as e:
            logger.exception("chunk_document_by_id_failed", error=str(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post(
    "/train/{document_id}",
    summary="Generate vector embeddings for a chunked document and save to MongoDB",
)
async def train_document_by_id(
    document_id: str,
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    stmt = select(Document).where(Document.id == document_id, Document.user_id == current_user.id)
    result = await session.execute(stmt)
    document = result.scalars().first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
        
    if document.status == DocumentStatus.untrained:
        raise HTTPException(status_code=400, detail="Document must be chunked before training")

    document.status = DocumentStatus.training
    await session.commit()
    
    async def event_generator():
        try:
            from app.agents.embedding_agent import EmbeddingAgent
            agent = EmbeddingAgent()
            
            total_chunks, remaining_chunks = await agent.get_remaining_chunks(document_id)
            
            if total_chunks == 0:
                yield f"data: {json.dumps({'error': 'No chunks found for this document. Please chunk it first.'})}\n\n"
                return
                
            if not remaining_chunks:
                yield f"data: {json.dumps({'progress': 100})}\n\n"
                document.status = DocumentStatus.trained
                document.training_progress = 100
                await session.commit()
                return

            # Batch process remaining chunks
            batch_size = agent.batch_size
            embedded_so_far = total_chunks - len(remaining_chunks)
            
            for i in range(0, len(remaining_chunks), batch_size):
                batch = remaining_chunks[i:i + batch_size]
                
                await agent.process_batch(document_id, batch)
                
                embedded_so_far += len(batch)
                progress_percentage = int((embedded_so_far / total_chunks) * 100)
                
                document.training_progress = progress_percentage
                await session.commit()
                
                yield f"data: {json.dumps({'progress': progress_percentage})}\n\n"

            document.status = DocumentStatus.trained
            await session.commit()
            
        except Exception as e:
            logger.exception("train_document_by_id_failed", error=str(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.delete(
    "/{document_id}",
    summary="Delete a document",
)
async def delete_document(
    document_id: str,
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    stmt = select(Document).where(Document.id == document_id, Document.user_id == current_user.id)
    result = await session.execute(stmt)
    document = result.scalars().first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # Remove file from disk
    if os.path.exists(document.file_path):
        os.remove(document.file_path)
        
    # Remove chunks and embeddings from MongoDB
    db = get_mongo_db()
    await db.document_chunks.delete_many({"document_id": document_id})
    await db.document_embeddings.delete_many({"document_id": document_id})

    # Remove from DB
    await session.delete(document)
    await session.commit()
    
    return {"message": "Document deleted successfully"}

