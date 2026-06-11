---
name: Agent Development
description: Guidelines for building agents, tools, and endpoints within this FastAPI + LangGraph starter project.
---

# Agentic AI Starter — FastAPI + LangGraph

This project is a boilerplate for creating API-driven AI Agents using FastAPI, LangChain, and LangGraph.

## Core Concepts

1. **Agents (`app/agents/`)**: Inherit from `BaseAgent` (`app/agents/base.py`). Agents encapsulate the LLM, the tools they use, and their specific LangGraph `StateGraph`.
2. **Tools (`app/tools/`)**: Functions decorated with `@tool` from `langchain_core.tools`.
3. **Routes (`app/api/routes/`)**: FastAPI endpoints that instantiate agents (usually via dependencies) and execute them, returning structured JSON.
4. **State**: The graph state flows between nodes (LLM node, tool execution node) using LangGraph.

## How to Add a New Agent

**1. Create your Tools (if any)**

```python
# app/tools/my_tool.py
from langchain_core.tools import tool

@tool
def my_tool(input: str) -> str:
    """Clear description for the LLM."""
    return f"Result for {input}"

MY_TOOLS = [my_tool]
```

**2. Create the Agent**

Create `app/agents/my_agent.py` and implement the `BaseAgent`:

```python
from typing import Any, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage
from typing_extensions import TypedDict, Annotated

from app.agents.base import BaseAgent
from app.tools.my_tool import MY_TOOLS
from app.core.logging import get_logger

logger = get_logger(__name__)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    metadata: dict

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(tools=MY_TOOLS, system_prompt="You are a helpful agent.")

    async def _call_llm(self, state: AgentState) -> AgentState:
        response = await self.llm.ainvoke(state["messages"])
        return {"messages": [response], "metadata": state.get("metadata", {})}

    async def _run_tools(self, state: AgentState) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        tool_map = {t.name: t for t in self.tools}
        results = []
        for call in tool_calls:
            output = await tool_map[call["name"]].ainvoke(call["args"])
            from langchain_core.messages import ToolMessage
            results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        return {"messages": results, "metadata": state.get("metadata", {})}

    def _should_continue(self, state: AgentState) -> Literal["run_tools", "__end__"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "run_tools"
        return "__end__"

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("call_llm", self._call_llm)
        graph.add_node("run_tools", self._run_tools)
        graph.add_edge(START, "call_llm")
        graph.add_conditional_edges("call_llm", self._should_continue,
                                    {"run_tools": "run_tools", "__end__": END})
        graph.add_edge("run_tools", "call_llm")
        return graph.compile()

    async def _run(self, user_input: str, thread_id: str = None) -> dict[str, Any]:
        initial_state = {
            "messages": [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_input),
            ],
            "metadata": {"thread_id": thread_id},
        }
        final_state = await self.graph.ainvoke(initial_state, {"recursion_limit": 10})
        return {
            "response": final_state["messages"][-1].content,
            "total_messages": len(final_state["messages"]),
        }
```

*Note on BaseAgent implementation*: `base.py` delegates to `self._run()` (which you must implement), but there may be a typo in the provided `base.py` calling `self.__run(...)`. Be aware of this when debugging!

**3. Wire into FastAPI**

```python
# app/api/routes/agent.py
from fastapi import APIRouter, Depends
from app.agents.my_agent import MyAgent

router = APIRouter()

def get_my_agent() -> MyAgent:
    return MyAgent()

@router.post("/my-agent/run")
async def run_my_agent(request: dict, agent: MyAgent = Depends(get_my_agent)):
    result = await agent.run(user_input=request["message"], thread_id=request.get("thread_id"))
    return result
```

## Logging and Configuration

- Always use `structlog` for logging. The project relies on it for structured JSON logs in production.
  ```python
  from app.core.logging import get_logger
  logger = get_logger(__name__)
  logger.info("event_name", key="value")
  ```
- Use `app.core.config.get_settings()` to load configuration variables. Ensure variables are present in `.env`.
