"""FastAPI application entry point.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.analyze import router as analyze_router

app = FastAPI(
    title="AI Debugging Assistant",
    description="LLM-powered error analysis and fix suggestion API.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return service health status for monitoring and readiness checks."""

    return {"status": "ok", "service": "ai-debug-assistant"}
