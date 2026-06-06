import logging
import structlog
from app.core.config import get_settings

settings = get_settings()


def setup_logging() -> None:
    """Call this once at app startup."""

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure standard library logging (uvicorn uses this)
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
    )

    # Shared processors — run on every log entry
    shared_processors = [
        structlog.contextvars.merge_contextvars,        # ← injects request_id etc.
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # JSON in production, pretty colors in development
    if settings.app_env == "production":
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=shared_processors,
        )
    else:
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
            foreign_pre_chain=shared_processors,
        )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Use this everywhere instead of logging.getLogger()."""
    return structlog.get_logger(name)