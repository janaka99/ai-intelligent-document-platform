import uuid
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

import structlog.contextvars

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.core.exceptions import http_exception_handler, unhandled_exception_handler
from app.api.routes import agent as agent_router
from app.api.routes import document as document_router
from app.api.routes import embedding as embedding_router

from app.db.database import engine, Base
from app.models.document import Document # Import to ensure it gets created
from app.core.users import fastapi_users_app, auth_backend
from app.schemas.user import UserRead, UserCreate, UserUpdate

settings = get_settings()
logger = get_logger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging()
    logger.info(
        "app_starting",
        name=settings.app_name,
        env=settings.app_env,
        version=settings.app_version,
    )
    
    # Create DB tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    yield
    # Shutdown
    logger.info("app_shutdown", name=settings.app_name)


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs" if settings.app_env != "production" else None,
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Add your frontend domains here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.include_router(agent_router.router, prefix="/api/v1", tags=["agents"])
app.include_router(document_router.router, prefix="/api/v1/document", tags=["documents"])

app.include_router(
    fastapi_users_app.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)
app.include_router(
    fastapi_users_app.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users_app.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)
# ── Middleware ─────────────────────────────────────────────────────────────────
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    # Clear context from previous request (important in async!)
    structlog.contextvars.clear_contextvars()

    # Bind a unique request_id — every log in this request carries it
    request_id = str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )

    start_time = time.perf_counter()
    logger.info("request_started")

    try:
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "request_finished",
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        # Send request_id back to client so they can reference it
        response.headers["X-Request-ID"] = request_id
        return response

    except Exception as exc:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.exception("request_failed", duration_ms=duration_ms, error=str(exc))
        raise


# ── Exception handlers ─────────────────────────────────────────────────────────
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, http_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)


# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(agent_router.router, prefix="/api/v1", tags=["agents"])
app.include_router(document_router.router, prefix="/api/v1/document", tags=["documents"])
app.include_router(embedding_router.router, prefix="/api/v1/embedding", tags=["embeddings"])

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "env": settings.app_env,
    }