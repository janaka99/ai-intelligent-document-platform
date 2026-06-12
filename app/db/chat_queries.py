from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, desc
import uuid

from app.models.chat import ChatSession, ChatMessage, ChatSummary, RoleEnum

async def create_session(session: AsyncSession, user_id: str, document_ids: list[str], title: str = "New Chat") -> ChatSession:
    chat_session = ChatSession(
        user_id=uuid.UUID(user_id),
        document_ids=document_ids,
        title=title
    )
    session.add(chat_session)
    await session.commit()
    await session.refresh(chat_session)
    return chat_session

async def get_session(session: AsyncSession, session_id: str) -> ChatSession | None:
    stmt = select(ChatSession).where(ChatSession.id == session_id, ChatSession.is_active == 1)
    result = await session.execute(stmt)
    return result.scalars().first()

async def get_session_by_document(session: AsyncSession, user_id: str, document_id: str) -> ChatSession | None:
    stmt = select(ChatSession).where(ChatSession.user_id == uuid.UUID(user_id), ChatSession.is_active == 1)
    result = await session.execute(stmt)
    sessions = result.scalars().all()
    for s in sessions:
        if document_id in s.document_ids:
            return s
    return None

async def list_sessions(session: AsyncSession, user_id: str) -> list[ChatSession]:
    stmt = select(ChatSession).where(ChatSession.user_id == uuid.UUID(user_id), ChatSession.is_active == 1).order_by(desc(ChatSession.created_at))
    result = await session.execute(stmt)
    return list(result.scalars().all())

async def add_message(session: AsyncSession, session_id: str, role: str, content: str, token_count: int = 0) -> ChatMessage:
    message = ChatMessage(
        session_id=session_id,
        role=RoleEnum(role),
        content=content,
        token_count=token_count
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)
    return message

async def get_messages(session: AsyncSession, session_id: str, limit: int = 50, before_id: str | None = None) -> list[ChatMessage]:
    # We order descending first to grab the latest N messages
    stmt = select(ChatMessage).where(ChatMessage.session_id == session_id)
    
    if before_id:
        # Get the creation time of the cursor message
        cursor_stmt = select(ChatMessage.created_at).where(ChatMessage.id == before_id)
        cursor_result = await session.execute(cursor_stmt)
        cursor_time = cursor_result.scalars().first()
        if cursor_time:
            stmt = stmt.where(ChatMessage.created_at < cursor_time)
            
    stmt = stmt.order_by(desc(ChatMessage.created_at)).limit(limit)
        
    result = await session.execute(stmt)
    messages = list(result.scalars().all())
    
    # We reverse the result so the messages are returned chronologically (oldest to newest)
    messages.reverse()
    return messages

async def increment_turn_count(session: AsyncSession, session_id: str) -> None:
    stmt = update(ChatSession).where(ChatSession.id == session_id).values(turn_count=ChatSession.turn_count + 1)
    await session.execute(stmt)
    await session.commit()

async def upsert_summary(session: AsyncSession, session_id: str, summary_text: str, covers_up_to_message_id: str | None = None) -> ChatSummary:
    stmt = select(ChatSummary).where(ChatSummary.session_id == session_id)
    result = await session.execute(stmt)
    existing_summary = result.scalars().first()
    
    if existing_summary:
        existing_summary.summary_text = summary_text
        existing_summary.covers_up_to_message_id = covers_up_to_message_id
    else:
        existing_summary = ChatSummary(
            session_id=session_id,
            summary_text=summary_text,
            covers_up_to_message_id=covers_up_to_message_id
        )
        session.add(existing_summary)
        
    await session.commit()
    await session.refresh(existing_summary)
    return existing_summary

async def get_active_summary(session: AsyncSession, session_id: str) -> ChatSummary | None:
    stmt = select(ChatSummary).where(ChatSummary.session_id == session_id)
    result = await session.execute(stmt)
    return result.scalars().first()
