"""FastAPI application factory with lifespan management."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nexus.config import config
from nexus.version import __version__


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    # ── Startup ──
    # TODO: Initialize:
    # - Database (init_db)
    # - Redis connection
    # - Load personas
    # - Register tools
    # - Create engine instance
    # Store on app.state for route access
    print(f"NEXUS v{__version__} starting...")
    yield
    # ── Shutdown ──
    # TODO: Close connections
    print("NEXUS shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="NEXUS",
        description="Notarized AI Execution — The agent framework where AI actions are accountable.",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # TODO: Add auth middleware
    # TODO: Mount route modules under /v1/ prefix

    # Import and include route modules
    from nexus.api.routes import execute, stream, ledger, personas, tools, knowledge, health, auth
    app.include_router(execute.router, prefix="/v1")
    app.include_router(stream.router, prefix="/v1")
    app.include_router(ledger.router, prefix="/v1")
    app.include_router(personas.router, prefix="/v1")
    app.include_router(tools.router, prefix="/v1")
    app.include_router(knowledge.router, prefix="/v1")
    app.include_router(health.router, prefix="/v1")
    app.include_router(auth.router, prefix="/v1")

    return app


app = create_app()
