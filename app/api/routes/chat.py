import json
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.db.database import get_async_session
from app.core.users import current_active_user
from app.models.user import User
from app.db.chat_queries import create_session, get_session, list_sessions, get_messages, get_session_by_document
from app.agents.chat_agent.graph import chat_agent_graph
from app.core.limiter import ChatLimiter
from app.core.token_counter import count_tokens
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()
limiter = ChatLimiter()

class CreateSessionRequest(BaseModel):
    document_id: str

class ChatMessageRequest(BaseModel):
    message: str

@router.post("/sessions", summary="Create a new chat session")
async def create_chat_session(
    request: CreateSessionRequest,
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    doc_id = request.document_id
    
    # Enforce 1 document = 1 chat rule
    existing_session = await get_session_by_document(db, str(user.id), doc_id)
    if existing_session:
        return {"session_id": existing_session.id}
        
    # Fetch the document to get its title
    from app.models.document import Document
    from sqlalchemy import select
    import uuid
    
    try:
        doc_uuid = uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
        
    stmt = select(Document).where(Document.id == doc_uuid, Document.user_id == user.id)
    result = await db.execute(stmt)
    document = result.scalars().first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
        
    session = await create_session(db, str(user.id), [doc_id], document.title)
    return {"session_id": session.id}

@router.get("/sessions", summary="List user's active chat sessions")
async def get_user_sessions(
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    sessions = await list_sessions(db, str(user.id))
    return [{"id": s.id, "title": s.title, "turn_count": s.turn_count, "created_at": s.created_at} for s in sessions]

@router.get("/sessions/{session_id}/messages", summary="Get messages for a session")
async def get_session_messages(
    session_id: str,
    before_id: str = None,
    limit: int = 50,
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    chat_session = await get_session(db, session_id)
    if not chat_session or str(chat_session.user_id) != str(user.id):
        raise HTTPException(status_code=404, detail="Session not found")
        
    messages = await get_messages(db, session_id, limit=limit, before_id=before_id)
    return [{"id": m.id, "role": m.role.value, "content": m.content, "created_at": m.created_at} for m in messages]

@router.post("/sessions/{session_id}/chat", summary="Chat with your documents via SSE stream")
async def chat_with_session(
    session_id: str,
    request: ChatMessageRequest,
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    chat_session = await get_session(db, session_id)
    if not chat_session or str(chat_session.user_id) != str(user.id):
        raise HTTPException(status_code=404, detail="Session not found")
        
    if not limiter.check_turn_limit(chat_session):
        raise HTTPException(status_code=429, detail="Maximum conversation turns reached for this session.")
        
    user_tokens = count_tokens(request.message)
    if not limiter.check_token_budget(user_tokens):
        raise HTTPException(status_code=413, detail="Your message is too long and exceeds the token budget.")
        
    initial_state = {
        "session_id": session_id,
        "user_message": request.message,
        "history": [],
        "retrieved_docs": [],
        "response": "",
        "token_usage": 0,
        "db_session": db,
        "chat_session": chat_session
    }

    async def event_generator():
        try:
            # We use astream_events to catch tokens as the LLM generates them inside the graph
            async for event in chat_agent_graph.astream_events(initial_state, version="v2"):
                kind = event["event"]
                name = event.get("name", "")
                
                # Print which node is currently executing to the terminal
                if kind == "on_chain_start" and name in ["memory", "agent", "tools", "persist"]:
                    logger.info(f"Executing LangGraph Node: {name.upper()}", session_id=session_id)
                
                # Stream when the search tool starts
                if kind == "on_tool_start" and name == "search_documents":
                    yield f"data: {json.dumps({'content': '\n\n*Searching documents...*\n\n'})}\n\n"
                    
                # Filter for LLM token chunks
                if kind == "on_chat_model_stream":
                    # Ensure we only stream the actual assistant response, not memory summaries
                    node_name = event.get("metadata", {}).get("langgraph_node")
                    if node_name == "agent":
                        chunk = event["data"]["chunk"]
                        # Do not stream if this chunk is part of a tool call
                        if chunk.content and not chunk.tool_calls:
                            # Yield Server-Sent Event formatted data
                            yield f"data: {json.dumps({'content': chunk.content})}\n\n"
                        
            yield f"data: {json.dumps({'done': True, 'turn_count': chat_session.turn_count + 1})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.delete("/sessions/{session_id}", summary="Soft delete a chat session")
async def delete_chat_session(
    session_id: str,
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    chat_session = await get_session(db, session_id)
    if not chat_session or str(chat_session.user_id) != str(user.id):
        raise HTTPException(status_code=404, detail="Session not found")
        
    # Soft delete
    chat_session.is_active = 0
    await db.commit()
    return {"status": "deleted"}
