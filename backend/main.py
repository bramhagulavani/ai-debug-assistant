"""FastAPI application entry point for AI Debugging Assistant.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.analyze import router as analyze_router
from backend.app.api.stream import router as stream_router
from backend.app.core.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup and shutdown logic around the app lifecycle."""

    await init_db()
    yield


app = FastAPI(
    title="AI Debugging Assistant",
    description="LLM-powered error analysis and fix suggestion API.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)
app.include_router(stream_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint that confirms the service is running."""

    return {"status": "ok", "service": "ai-debug-assistant"}
