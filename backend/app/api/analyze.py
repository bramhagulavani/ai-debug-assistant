"""Analyze endpoint that orchestrates parsing and LLM response generation."""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.prompts.debug_prompt import build_debug_system_prompt, build_debug_user_prompt
from backend.app.services.ast_parser import ASTParserService
from backend.app.services.llm_service import LLMService, LLMMessage, LLMServiceError
from backend.app.services.stack_trace_parser import StackTraceParser

router = APIRouter(prefix="/api", tags=["analyze"])

stack_trace_parser = StackTraceParser()
ast_parser_service = ASTParserService()
llm_service: Optional[LLMService] = None


def _get_llm_service() -> LLMService:
    """Return a shared LLM service instance, creating it lazily on first use."""

    global llm_service
    if llm_service is None:
        try:
            llm_service = LLMService()
        except Exception as exc:  # noqa: BLE001
            raise LLMServiceError(f"Failed to initialize LLM service: {exc}") from exc
    return llm_service


class AnalyzeRequest(BaseModel):
    """Incoming payload for stack-trace-guided code analysis."""

    code: str = Field(..., min_length=1)
    stack_trace: str = Field(..., min_length=1)
    filename: Optional[str] = None
    language: Optional[str] = None


class AnalyzeResponse(BaseModel):
    """Structured API response returned to the client."""

    success: bool
    language: str
    error_type: str
    error_message: str
    error_line: Optional[int]
    error_function: Optional[str]
    ai_response: str
    duration_ms: int


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_error(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze code and stack trace and return an LLM-generated fix explanation."""

    start = time.monotonic()

    parsed_trace = stack_trace_parser.parse(request.stack_trace)
    parsed_context = ast_parser_service.parse(
        code=request.code,
        error_line=parsed_trace.error_line,
        filename=request.filename,
        language=request.language,
    )

    system_prompt = build_debug_system_prompt()
    user_prompt = build_debug_user_prompt(
        stack_trace_context=parsed_trace.to_prompt_string(),
        ast_context=parsed_context.to_prompt_string(),
        raw_code=request.code,
    )

    try:
        llm_response = await _get_llm_service().generate_response(
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            max_tokens=2000,
        )
    except LLMServiceError as exc:
        raise HTTPException(status_code=503, detail=f"LLM service unavailable: {exc}") from exc

    duration_ms = int((time.monotonic() - start) * 1000)

    return AnalyzeResponse(
        success=True,
        language=parsed_trace.language,
        error_type=parsed_trace.error_type,
        error_message=parsed_trace.error_message,
        error_line=parsed_trace.error_line,
        error_function=parsed_trace.error_function,
        ai_response=llm_response.content,
        duration_ms=duration_ms,
    )


__all__ = ["router", "AnalyzeRequest", "AnalyzeResponse", "analyze_error"]
