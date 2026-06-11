from typing import Annotated, Any, Dict, List, Optional, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage

from app.agents.base import BaseAgent
from app.core.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an AI Document Intelligence Agent.
Your job is to analyze the provided document content, extract relevant information, and answer user queries.
Use the available tools to process the document when needed.
Always be concise and clear in your responses."""

class DocumentState(TypedDict):
    # Standard LangGraph conversation history
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Core Document Info
    document_id: str
    file_name: str
    file_type: str
    
    # Processing state
    raw_content: Optional[str]
    extracted_data: Dict[str, Any]
    summary: Optional[str]
    
    # Workflow control
    status: str
    errors: List[str]
    metadata: Dict[str, Any]


class DocumentAgent(BaseAgent):
    def __init__(self, tools: list = []):
        super().__init__(tools=tools, system_prompt=SYSTEM_PROMPT)

    async def _call_llm(self, state: DocumentState) -> DocumentState:
        """Node 1: Send messages to the LLM and get a response."""
        logger.info("node_call_llm", message_count=len(state["messages"]), document_id=state.get("document_id"))
        
        # We can inject document context into the LLM call if needed
        # For now, just pass the standard messages
        response = await self.llm.ainvoke(state["messages"])
        return {"messages": [response], "metadata": state.get("metadata", {})}

    async def _run_tools(self, state: DocumentState) -> DocumentState:
        """Node 2: execute every tool call the LLM requested."""
        tool_calls = state["messages"][-1].tool_calls
        tool_map = {t.name: t for t in self.tools}

        tool_results = []   

        for call in tool_calls:
            tool_name = call["name"]
            logger.info("tool_called", tool_name=tool_name, arguments=call["args"], document_id=state.get("document_id"))
            
            if tool_name not in tool_map:
                output = f"Error: tool '{tool_name}' not found."
            else:
                try:
                    # Pass document context to tools if they need it
                    output = await tool_map[tool_name].ainvoke(call["args"])
                except Exception as e:
                    output = f"Error running {tool_name}: {e}"
                    logger.error("tool_failed", tool=tool_name, error=str(e))

            tool_results.append(
                ToolMessage(content=str(output), tool_call_id=call["id"])
            )
        
        return {"messages": tool_results, "metadata": state.get("metadata", {})}

    def _should_continue(self, state: DocumentState) -> Literal["run_tools", "__end__"]:
        """Conditional edge: does the last message have tool calls?"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "run_tools"
        return "__end__"

    def _build_graph(self):
        # register nodes
        graph = StateGraph(DocumentState)

        # Add nodes
        graph.add_node("call_llm", self._call_llm)
        graph.add_node("run_tools", self._run_tools)

        # add edges
        graph.add_edge(START, "call_llm")

        # Conditional edge
        graph.add_conditional_edges(
            "call_llm",
            self._should_continue,
            ["run_tools", "__end__"]
        )

        # After tools run, always go back to LLM
        graph.add_edge("run_tools", "call_llm") 

        return graph.compile()
    
    async def _run(self, user_input: str, thread_id: str = None, document_id: str = None, raw_content: str = None) -> dict[str, Any]:
        """Run the agent using the compiled graph."""
        
        # If we have raw content, we inject it as context in the first message
        context = ""
        if raw_content:
            context = f"\n\nDocument Context:\n{raw_content}"
            
        initial_state = {
            "messages": [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_input + context),
            ],
            "document_id": document_id or "unknown",
            "file_name": "unknown",
            "file_type": "text/plain",
            "raw_content": raw_content,
            "extracted_data": {},
            "status": "processing",
            "errors": [],
            "metadata": {"thread_id": thread_id}
        }

        final_state = await self.graph.ainvoke(initial_state, {"recursion_limit": 10})

        # Extract the final text response
        last_message = final_state["messages"][-1]
        return {
            "response": last_message.content,
            "total_messages": len(final_state["messages"]),
            "extracted_data": final_state.get("extracted_data", {})
        }
