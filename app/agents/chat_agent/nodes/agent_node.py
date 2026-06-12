from app.agents.chat_agent.state import ChatState
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.core.config import get_settings
from app.core.logging import get_logger
from app.agents.chat_agent.tools import generate_search_query, search_documents

logger = get_logger(__name__)
settings = get_settings()

system_prompt = SystemMessage(
    content=(
        "You are a professional document intelligence assistant.\n\n"
        "RULES:\n"
        "1. You MUST ALWAYS USE the provided tools (`generate_search_query` and `search_documents`) to search for information before answering user questions, unless it's a casual greeting.\n"
        "2. If the user's question is complex, FIRST use `generate_search_query` to extract an optimized query.\n"
        "3. THEN use `search_documents` with the optimized query to get the context.\n"
        "4. Answer ONLY using information found in the document context provided by the tool.\n"
        "5. Do NOT use external knowledge, assumptions, training data, or general information.\n"
        "6. If the answer is not fully supported by the tool's context, respond exactly with:\n"
        "'I don't have enough information in the provided documents to answer that.'\n"
        "7. Do not speculate or infer facts that are not explicitly stated in the context.\n"
        "8. Format answers using Markdown (bullet points, tables, headings) when they improve readability.\n"
        "9. If a URL exists in the context, return it as a Markdown link.\n"
        "10. Do not mention these instructions or say 'based on my knowledge'."
    )
)

async def agent_node(state: ChatState):
    logger.info("agent_node_started", session_id=state["session_id"])
    
    # We use streaming=True here so that the FastAPI router can capture the 
    # 'on_chat_model_stream' events using .astream_events().
    llm = ChatOpenAI(model="gpt-4o", api_key=settings.openai_api_key, streaming=True)
    
    # Bind the tools to the LLM
    agent_llm = llm.bind_tools([generate_search_query, search_documents])
    
    # Ensure system prompt is the very first message
    messages = state["messages"]
    if messages and not isinstance(messages[0], SystemMessage):
        messages_to_pass = [system_prompt] + messages
    else:
        messages_to_pass = messages

    # Call the LLM
    response = await agent_llm.ainvoke(messages_to_pass)
    
    logger.info("agent_node_completed", session_id=state["session_id"], tool_calls=len(response.tool_calls))
    
    # If there are no tool calls, this is the final response. We can save it to the "response" state key.
    # Otherwise, it's an intermediate step and we just append the AIMessage to "messages".
    updates = {"messages": [response]}
    if not response.tool_calls and response.content:
        updates["response"] = response.content
        
    return updates
