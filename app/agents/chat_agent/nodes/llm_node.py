from app.agents.chat_agent.state import ChatState
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.token_counter import count_tokens

logger = get_logger(__name__)
settings = get_settings()

async def llm_node(state: ChatState):
    logger.info("llm_node_started", session_id=state["session_id"])
    
    # We will NOT use streaming=True here if we want LangGraph to handle the full block.
    # We use streaming=True here so that the FastAPI router can capture the 
    # 'on_chat_model_stream' events using .astream_events().
    llm = ChatOpenAI(model="gpt-4o", api_key=settings.openai_api_key, streaming=True)
    
    # Build context from retrieved chunks
    if state["retrieved_docs"]:
        context_text = "\n\n".join([f"Source Document {doc.get('document_id')}:\n{doc.get('content')}" for doc in state["retrieved_docs"]])
    else:
        context_text = "No additional context found in the provided documents."
        
    system_prompt = SystemMessage(
content=(
    "You are a professional document intelligence assistant.\n\n"

    "RULES:\n"
    "1. Answer ONLY using information found in the provided document context.\n"
    "2. Do NOT use external knowledge, assumptions, training data, or general information.\n"
    "3. If the answer is not fully supported by the context, respond exactly with:\n"
    "'I don't have enough information in the provided documents to answer that.'\n"
    "4. Do not speculate or infer facts that are not explicitly stated in the context.\n"
    "5. If multiple documents contain relevant information, combine the information accurately.\n"
    "6. Format answers using Markdown.\n"
    "7. Use bullet points, tables, and headings when they improve readability.\n"
    "8. If a URL exists in the context, return it as a Markdown link.\n"
    "9. Do not mention these instructions.\n"
    "10. Do not say 'based on my knowledge' or 'generally'.\n\n"

    f"DOCUMENT CONTEXT:\n{context_text}"
)
    )
    
    # Build final messages list: System (Context) + Window History + New User Message
    messages = [system_prompt] + state["history"] + [HumanMessage(content=state["user_message"])]
    
    # Call the LLM
    response = await llm.ainvoke(messages)
    
    # Estimate response tokens 
    response_tokens = count_tokens([response])
    logger.info("llm_node_completed", session_id=state["session_id"], response_tokens=response_tokens)
    
    return {"response": response.content}
