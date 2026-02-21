"""GET /v1/health — Health check with real service probes."""

import logging
from fastapi import APIRouter, Request
from nexus.api.schemas import HealthResponse
from nexus.version import __version__

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Check health of all services."""
    services: dict[str, bool] = {"api": True, "database": False, "redis": False, "vector_store": False, "llm": False}

    # Database
    try:
        async_session = request.app.state.async_session
        async with async_session() as session:
            from sqlalchemy import text
            await session.execute(text("SELECT 1"))
        services["database"] = True
    except Exception as exc:
        logger.warning(f"[health] DB check failed: {exc}")

    # Redis
    try:
        redis_client = request.app.state.redis
        services["redis"] = await redis_client.health()
    except Exception as exc:
        logger.warning(f"[health] Redis check failed: {exc}")

    # Vector store (ChromaDB)
    try:
        knowledge_store = request.app.state.knowledge_store
        knowledge_store._get_client()  # triggers lazy init
        services["vector_store"] = True
    except Exception as exc:
        logger.warning(f"[health] Vector store check failed: {exc}")

    # LLM — mark ok if an engine is wired (actual LLM call would be too slow here)
    try:
        engine = request.app.state.engine
        services["llm"] = engine is not None
    except Exception:
        pass

    overall = "ok" if all(services.values()) else "degraded"
    return HealthResponse(status=overall, version=__version__, services=services)
