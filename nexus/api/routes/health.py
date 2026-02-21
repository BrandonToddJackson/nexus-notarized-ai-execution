"""GET /v1/health — Health check."""

from fastapi import APIRouter
from nexus.api.schemas import HealthResponse
from nexus.version import __version__

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health of all services.

    Returns status of: app, database, redis, vector store, LLM provider.
    """
    # TODO: Check each service
    return HealthResponse(
        status="ok",
        version=__version__,
        services={
            "api": True,
            "database": False,  # TODO: check
            "redis": False,     # TODO: check
            "vector_store": False,  # TODO: check
            "llm": False,       # TODO: check
        }
    )
