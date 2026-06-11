import uuid
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from app.agents.example_agent import ExampleAgent
from app.core.logging import get_logger
from app.core.users import current_active_user
from app.models.user import User

router = APIRouter()
logger = get_logger(__name__)

# ── Request / Response models ──────────────────────────────────────────────────

class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    thread_id: str | None = Field(default=None, description="Optional session ID")

    model_config = {"json_schema_extra": {"example": {"message": "What time is it?"}}}


class AgentResponse(BaseModel):
    response: str
    thread_id: str
    total_messages: int


class ErrorResponse(BaseModel):
    error: str
    path: str

def get_example_agent() -> ExampleAgent:
    """
    FastAPI calls this to inject the agent into routes.
    Swap ExampleAgent for any other agent here — routes don't change.
    """
    return ExampleAgent()

@router.post(
    "/agent/run",
    response_model=AgentResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Run the example agent",
)
async def run_agent(
    request: AgentRequest,
    agent: ExampleAgent = Depends(get_example_agent),
    current_user: User = Depends(current_active_user),
):
    thread_id = request.thread_id or str(uuid.uuid4())

    logger.info(
        "route_agent_run",
        thread_id=thread_id,
        message_preview=request.message[:60],
    )

    try:
        result = await agent._run(
            user_input=request.message,
            thread_id=thread_id,
        )
        return AgentResponse(
            response=result["response"],
            thread_id=thread_id,
            total_messages=result["total_messages"],
        )

    except Exception as e:
        logger.exception("route_agent_failed", thread_id=thread_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent/info", summary="Get agent capabilities")
async def agent_info(
    agent: ExampleAgent = Depends(get_example_agent),
    current_user: User = Depends(current_active_user),
):
    return {
        "agent": agent.__class__.__name__,
        "model": agent.llm.model_name if hasattr(agent.llm, "model_name") else "unknown",
        "tools": [
            {"name": t.name, "description": t.description}
            for t in agent.tools
        ],
    }