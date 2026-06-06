from app.core import config
from typing import Any, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from typing_extensions import TypedDict, Annotated

from app.agents.base import BaseAgent
from app.tools.example_tool import EXAMPLE_TOOLS
from app.core.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
Use tools when they help answer the user's question accurately.
Always be concise and clear in your responses."""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    metadata: dict 


class ExampleAgent(BaseAgent):
    def __init__(self):
        super().__init__(tools=EXAMPLE_TOOLS,system_prompt=SYSTEM_PROMPT)

    async def _call_llm(self,state:AgentState) -> AgentState:
        """Node 1: send messages to the LLM, get a response."""
        logger.info("node_call_llm", message_count=len(state["messages"]))

        response = await self.llm.ainvoke(state["messages"])

        return {"messages": [response], "metadata": state.get("metadata", {})}
        # 
    
    async def _run_tools(self,state:AgentState) -> AgentState:
        """Node 2: execute every tool call the LLM requested."""
        tool_calls = state["messages"][-1].tool_calls
        tool_map = {t.name: t for t in self.tools}

        tool_results = []   

        for call in tool_calls:
            tool_name = call["name"]
            logger.info("tool_called", tool_name=tool_name, arguments=call["args"])
            
            if tool_name not in tool_map:
                output = f"Error: tool '{tool_name}' not found."
            else:
                try:
                    output = await tool_map[tool_name].ainvoke(call["args"])
                except Exception as e:
                    output = f"Error running {tool_name}: {e}"
                    logger.error("tool_failed", tool=tool_name, error=str(e))

            tool_results.append(
                ToolMessage(content=str(output), tool_call_id=call["id"])
            )
        
        return {"messages": tool_results, "metadata": state.get("metadata",{})}

    def _should_continue(self, state: AgentState) -> Literal["run_tools", "__end__"]:
        """Conditional edge: does the last message have tool calls?"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "run_tools"
        return "__end__"
                        

    def _build_graph(self):

        # register nodes
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("call_llm",self._call_llm)
        graph.add_node("run_tools",self._run_tools)

        # add edges
        graph.add_edge(START,"call_llm")

        # Conditional edge
        graph.add_conditional_edges(
            "call_llm",
            self._should_continue,
            ["run_tools","__end__"]
        )

        # After tools run, always go back to LLM
        graph.add_edge("run_tools", "call_llm") 

        return graph.compile()
    
    async def _run(self,user_input:str,thread_id:str=None) -> dict[str,Any]:
        """Run the agent using the compiled graph."""
        
        initial_state = {
            "messages": [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_input),
            ],
        }
        config = {"recursion_limit": 10 }

        final_state = await self.graph.ainvoke(initial_state)

         # Extract the final text response
        last_message = final_state["messages"][-1]
        return {
            "response": last_message.content,
            "total_messages": len(final_state["messages"]),
        }