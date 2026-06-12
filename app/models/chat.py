import uuid
import enum
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Enum, JSON, UUID
from sqlalchemy.orm import relationship

from app.db.database import Base

class RoleEnum(str, enum.Enum):
    user = "user"
    assistant = "assistant"

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    
    # SQLite doesn't natively support arrays, so we use JSON for document_ids
    document_ids = Column(JSON, default=list, nullable=False)
    
    title = Column(String, nullable=True)
    turn_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    is_active = Column(Integer, default=1, nullable=False) # Soft delete flag

    user = relationship("User", backref="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    summaries = relationship("ChatSummary", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(Enum(RoleEnum), nullable=False)
    content = Column(String, nullable=False)
    token_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    session = relationship("ChatSession", back_populates="messages")

class ChatSummary(Base):
    __tablename__ = "chat_summaries"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    summary_text = Column(String, nullable=False)
    covers_up_to_message_id = Column(String, ForeignKey("chat_messages.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    session = relationship("ChatSession", back_populates="summaries")
