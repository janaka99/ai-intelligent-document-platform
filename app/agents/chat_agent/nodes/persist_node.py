from app.agents.chat_agent.state import ChatState
from app.db.chat_queries import add_message, increment_turn_count
from app.core.token_counter import count_tokens
from app.core.logging import get_logger

logger = get_logger(__name__)

async def persist_node(state: ChatState):
    logger.info("persist_node_started", session_id=state["session_id"])
    
    db_session = state["db_session"]
    session_id = state["session_id"]
    
    # Count tokens for persistence
    user_tokens = count_tokens(state["user_message"])
    assistant_tokens = count_tokens(state["response"])
    
    # Persist the user message to the database
    await add_message(db_session, session_id, "user", state["user_message"], user_tokens)
    
    # Persist the assistant response to the database
    await add_message(db_session, session_id, "assistant", state["response"], assistant_tokens)
    
    # Increment the session turn count
    await increment_turn_count(db_session, session_id)
    
    total_tokens = user_tokens + assistant_tokens
    logger.info("persist_node_completed", session_id=session_id, total_turn_tokens=total_tokens)
    
    return {"token_usage": total_tokens}
