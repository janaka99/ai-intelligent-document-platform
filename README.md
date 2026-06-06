# 🤖 Agentic AI Starter — FastAPI + LangGraph

A production-ready starter kit for building AI agents with FastAPI and LangGraph. Handles all the boilerplate — logging, config, error handling, middleware — so you can focus entirely on building agents.

---

## ✨ Features

- **FastAPI** — async HTTP layer with auto-generated Swagger docs
- **LangGraph** — stateful agent graphs with tool calling and conditional routing
- **LangChain** — swappable LLM providers (OpenAI, Anthropic, Ollama)
- **Structured logging** — JSON logs with automatic `request_id` tracing via `structlog`
- **Pydantic Settings** — type-safe config loaded from `.env`
- **Global error handling** — clean JSON error responses, full tracebacks in logs
- **Dependency injection** — agents injected into routes, easy to swap
- **Production folder structure** — simple, flat, and easy to navigate

---

## 📁 Project Structure

```
agentic-starter/
├── app/
│   ├── main.py                  # FastAPI app, middleware, lifespan, routes
│   ├── core/
│   │   ├── config.py            # All settings loaded from .env (Pydantic)
│   │   ├── logging.py           # Structured JSON logger setup (structlog)
│   │   └── exceptions.py        # Global HTTP & unhandled exception handlers
│   ├── agents/
│   │   ├── base_agent.py        # Reusable base class every agent extends
│   │   └── example_agent.py     # Your first working LangGraph agent
│   ├── api/
│   │   └── routes/
│   │       └── agent.py         # HTTP routes — request/response models + endpoints
│   └── tools/
│       └── example_tool.py      # Custom tools agents can call (@tool decorated)
├── .env                         # Secrets & config (never commit this)
├── .env.example                 # Safe template to commit
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone & create virtual environment

```bash
git clone https://github.com/your-username/agentic-starter.git
cd agentic-starter

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```env
APP_NAME="Agentic Starter"
APP_ENV="development"
APP_VERSION="0.1.0"
LOG_LEVEL="INFO"

# LLM
OPENAI_API_KEY="sk-..."

```

### 4. Run the server

```bash
uvicorn app.main:app --reload
```

Expected output:
```
[info] app_starting  name=Agentic Starter  env=development  version=0.1.0
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | App health, version, environment |
| `GET` | `/docs` | Swagger UI (disabled in production) |
| `GET` | `/api/v1/agent/info` | Agent name, model, and available tools |
| `POST` | `/api/v1/agent/run` | Run the agent with a user message |

### POST `/api/v1/agent/run`

**Request:**
```json
{
  "message": "What is the current UTC time?",
  "thread_id": "optional-session-id"
}
```

**Response:**
```json
{
  "response": "The current UTC time is 2024-01-15 10:23:01 UTC.",
  "thread_id": "abc-123",
  "total_messages": 4
}
```

**Quick test with curl:**
```bash
# Plain conversation
curl -X POST http://localhost:8000/api/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"message": "What can you help me with?"}'

# Trigger a tool call
curl -X POST http://localhost:8000/api/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the current UTC time?"}'

# With a sticky session
curl -X POST http://localhost:8000/api/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, my name is Alex", "thread_id": "session-001"}'
```

---

## 🧠 How LangGraph Agents Work

Every agent follows the same pattern: **state flows through nodes, a router decides what happens next.**

```
START
  ↓
call_llm  ──→  should_continue? ──→ (no tool calls) ──→ END
                     ↓
               (has tool calls)
                     ↓
               run_tools ──→ call_llm  (loop)
```

### Core concepts

| Concept | What it is |
|---------|------------|
| **State** | A `TypedDict` that flows through the graph. Use `add_messages` reducer to accumulate conversation history automatically. |
| **Node** | An async function: receives state → returns updated state. One job per node. |
| **Router** | A plain function returning an edge name string. Controls flow via `add_conditional_edges`. |

---

## 🔧 Adding a New Agent

Three steps — that's it. The boilerplate never changes.

### Step 1 — Create the agent

```python
# app/agents/my_agent.py
from typing import Any, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage
from typing_extensions import TypedDict, Annotated

from app.agents.base_agent import BaseAgent
from app.tools.my_tools import MY_TOOLS
from app.core.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = "You are a specialist agent that does X."


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    metadata: dict


class MyAgent(BaseAgent):

    def __init__(self):
        super().__init__(tools=MY_TOOLS, system_prompt=SYSTEM_PROMPT)

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
        final_state = await self.graph.ainvoke(
            initial_state, {"recursion_limit": 10}
        )
        return {
            "response": final_state["messages"][-1].content,
            "total_messages": len(final_state["messages"]),
        }
```

### Step 2 — Add a dependency

```python
# In app/api/routes/agent.py
from app.agents.my_agent import MyAgent

def get_my_agent() -> MyAgent:
    return MyAgent()
```

### Step 3 — Add a route

```python
@router.post("/my-agent/run", response_model=AgentResponse)
async def run_my_agent(
    request: AgentRequest,
    agent: MyAgent = Depends(get_my_agent),
):
    thread_id = request.thread_id or str(uuid.uuid4())
    result = await agent.run(user_input=request.message, thread_id=thread_id)
    return AgentResponse(
        response=result["response"],
        thread_id=thread_id,
        total_messages=result["total_messages"],
    )
```

---

## 🛠️ Adding a New Tool

```python
# app/tools/my_tools.py
from langchain_core.tools import tool


@tool
def my_tool(input: str) -> str:
    """Clear description of what this tool does — the LLM reads this."""
    return f"Result for: {input}"


MY_TOOLS = [my_tool]
```

> **Tip:** The docstring is what the LLM uses to decide when to call the tool. Make it specific and clear.

---

## 📋 Logging

Use the structured logger anywhere in the codebase:

```python
from app.core.logging import get_logger

logger = get_logger(__name__)

logger.info("event_name", key="value", another_key=123)
logger.warning("token_limit_close", tokens_used=3800, limit=4000)
logger.error("tool_failed", tool="web_search", error=str(e))
logger.exception("unhandled_error", context="agent_run")  # includes stack trace
```

Every log line automatically includes `request_id`, `method`, and `path` from the middleware — no extra work needed.

**Development output** (colored):
```
2024-01-15 10:23:01 [info] agent_run_started  agent=ExampleAgent thread_id=abc-123
```

**Production output** (JSON):
```json
{"event": "agent_run_started", "agent": "ExampleAgent", "thread_id": "abc-123", "level": "info", "timestamp": "2024-01-15T10:23:01Z", "request_id": "xyz-456"}
```

Switch to JSON by setting `APP_ENV=production` in your `.env`.

---

## ⚙️ Configuration Reference

All settings live in `app/core/config.py` and are loaded from `.env`.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `APP_NAME` | `str` | `"Agentic Starter"` | Application name |
| `APP_ENV` | `str` | `"development"` | Environment (`development` / `production`) |
| `APP_VERSION` | `str` | `"0.1.0"` | App version shown in `/health` |
| `LOG_LEVEL` | `str` | `"INFO"` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `OPENAI_API_KEY` | `str` | *(required)* | Your OpenAI API key |
| `LLM_MODEL` | `str` | `"gpt-4o-mini"` | Model name passed to ChatOpenAI |
| `LLM_TEMPERATURE` | `float` | `0.0` | LLM temperature (0 = deterministic) |
| `AGENT_MAX_ITERATIONS` | `int` | `10` | Max LangGraph recursion depth |
| `AGENT_TIMEOUT_SECONDS` | `int` | `60` | Agent run timeout |

To add a new setting:
1. Add it to `.env` and `.env.example`
2. Add a typed field to `Settings` in `app/core/config.py`
3. Access it anywhere via `get_settings().your_field`

---

## 🐛 Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ValidationError: openai_api_key missing` | `.env` not found or key missing | Ensure `.env` is in the project root (same level as `app/`) |
| `ModuleNotFoundError: app` | Running uvicorn from wrong directory | Run from `agentic-starter/`, not inside `app/` |
| `RecursionError` in LangGraph | Agent looping without reaching END | Check `_should_continue` — ensure it can return `"__end__"` |
| `Tool not found` | Tool name mismatch | Tool name comes from the function name, not the variable name |
| `422 Unprocessable Entity` | Bad request body | Ensure `message` field is present and non-empty |
| Swagger UI not loading | Running in production mode | Set `APP_ENV=development` in `.env` |

---

## 🗺️ What to Build Next

Once comfortable with the starter, these are natural extensions:

- **Streaming responses** — swap `ainvoke` for `astream_events` + FastAPI `StreamingResponse`
- **Conversation memory** — add LangGraph `MemorySaver` checkpointing for multi-turn sessions
- **More tools** — web search (`TavilySearch`), code execution, database queries
- **Multi-agent** — a supervisor agent that routes to specialist sub-agents
- **Auth** — add `HTTPBearer` dependency to protect routes
- **Docker** — containerise with a simple `Dockerfile` + `docker-compose.yml`
- **Tests** — use `httpx.AsyncClient` with FastAPI's `TestClient` for route testing

---

## 📦 Dependencies

```
fastapi
uvicorn[standard]
langchain
langgraph
langchain-openai
python-dotenv
pydantic-settings
structlog
httpx
```

---

## 📄 License

MIT — use it however you like.