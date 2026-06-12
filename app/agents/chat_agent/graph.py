from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from app.agents.chat_agent.state import ChatState
from app.agents.chat_agent.nodes.memory_node import memory_node
from app.agents.chat_agent.nodes.agent_node import agent_node
from app.agents.chat_agent.nodes.persist_node import persist_node
from app.agents.chat_agent.tools import generate_search_query, search_documents

def route_tools(state: ChatState):
    """
    Conditional edge to route between the tools and persist node.
    """
    messages = state.get("messages", [])
    if not messages:
        return "persist"
        
    last_message = messages[-1]
    # If the LLM makes a tool call, route to tools
    if last_message.tool_calls:
        return "tools"
    # Otherwise, it's the final answer
    return "persist"

def build_chat_graph():
    """
    Compiles the tool-calling state graph for the document chat feature.
    State flows: Memory -> Agent <-> Tools
                             |
                           Persist
    """
    graph = StateGraph(ChatState)
    
    # Initialize the ToolNode with our tools
    tool_node = ToolNode([generate_search_query, search_documents])
    
    # Add nodes
    graph.add_node("memory", memory_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("persist", persist_node)
    
    # Add edges
    graph.set_entry_point("memory")
    graph.add_edge("memory", "agent")
    
    # Conditional routing from agent
    graph.add_conditional_edges(
        "agent",
        route_tools,
        {
            "tools": "tools",
            "persist": "persist"
        }
    )
    
    # After tools run, always return to the agent to evaluate the results
    graph.add_edge("tools", "agent")
    
    # End from persist
    graph.add_edge("persist", END)
    
    return graph.compile()

# This is the compiled agent that routes will invoke
chat_agent_graph = build_chat_graph()
