"""FastAPI application entrypoint for AI Debugging Assistant."""

from __future__ import annotations

from fastapi import FastAPI

from backend.app.api.analyze import router as analyze_router
from backend.app.api.auth import router as auth_router
from backend.app.api.stream import router as stream_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""

    app = FastAPI(
        title="AI Debugging Assistant API",
        version="0.1.0",
        description="Analyzes stack traces and source code using LLM orchestration.",
    )

    @app.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        """Return a simple health status for uptime checks."""

        return {"status": "ok"}

    app.include_router(analyze_router)
    app.include_router(stream_router)
    app.include_router(auth_router)
    return app


app = create_app()
