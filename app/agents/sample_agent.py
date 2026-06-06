from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,END
from app.core.logging import logger
from app.core.config import get_settings



settings = get_settings()
openai_api_key = settings.OPENAI_API_KEY

# 1.  Define the state that passed through the graph
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]


# 2. Initialize the LLM using our validated config
llm = ChatOpenAI(model="gpt-4o-mini",api_key=openai_api_key)

# 3. Define node function
def call_model(state: AgentState):
    """Calls the LLM and updates the state with the response."""
    logger.info("Agent node 'call_model' triggered.")
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# 4. Compile the graph
workflow = StateGraph(AgentState)

# 5. Add out node to graph
workflow.add_node("call_model",call_model)

workflow.set_entry_point("call_model")
workflow.add_edge("call_model",END)

agent_executor = workflow.compile()