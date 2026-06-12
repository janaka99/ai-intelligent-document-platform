from app.agents.chat_agent.state import ChatState
from app.memory.manager import MemoryManager
from app.core.config import get_settings
from langchain_openai import ChatOpenAI
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

async def memory_node(state: ChatState):
    logger.info("memory_node_started", session_id=state["session_id"])
    
    # We use a cheaper model to compress the summaries if needed
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
    
    manager = MemoryManager(
        window_size=settings.window_size,
        summarize_threshold=settings.summarize_threshold,
        token_budget=settings.token_budget,
        llm=llm
    )
    
    history = await manager.build_context(state["db_session"], state["session_id"])
    
    from langchain_core.messages import HumanMessage
    initial_messages = history + [HumanMessage(content=state["user_message"])]
    
    logger.info("memory_node_completed", session_id=state["session_id"], history_length=len(history))
    return {"history": history, "messages": initial_messages}
