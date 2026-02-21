"""Async SQLAlchemy engine and session factory."""

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from nexus.config import config

engine = create_async_engine(config.database_url, echo=config.debug)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncSession:
    """Dependency injection for FastAPI routes."""
    async with async_session() as session:
        yield session


async def init_db():
    """Create all tables. Called at startup."""
    from nexus.db.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
