"""Database connection and session management.

Provides:
    engine       - SQLAlchemy async engine connected to PostgreSQL
    AsyncSession - async session factory for database operations
    get_db()     - FastAPI dependency that yields a session per request
    init_db()    - creates all tables on startup (dev only)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.app.core.config import settings
from backend.app.models.database import Base

_db_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")

engine = create_async_engine(
    _db_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield one database session per request and handle commit/rollback."""

    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create all model tables. Use Alembic in production deployments."""

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
