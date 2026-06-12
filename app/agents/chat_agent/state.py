from typing import TypedDict, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.chat import ChatSession

class ChatState(TypedDict):
    session_id: str
    user_message: str
    history: list[BaseMessage]
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_docs: list[dict]
    response: str
    token_usage: int
    
    # Database dependencies passed through state
    db_session: AsyncSession
    chat_session: ChatSession
