"""API routes for stack-trace-driven code analysis."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.models.analysis import (
    AnalysisResult,
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorLocation,
)
from backend.app.prompts.analysis_prompt import (
    ANALYZE_SYSTEM_PROMPT,
    build_analysis_user_prompt,
)
from backend.app.services.ast_parser import ASTParserService
from backend.app.services.llm_service import LLMMessage, LLMService, LLMServiceError
from backend.app.services.stack_trace_parser import StackTraceParser

router = APIRouter(prefix="/api", tags=["analysis"])

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def get_stack_trace_parser() -> StackTraceParser:
    """Return a stack trace parser dependency instance."""

    return StackTraceParser()


def get_ast_parser_service() -> ASTParserService:
    """Return an AST parser dependency instance."""

    return ASTParserService()


def get_llm_service() -> LLMService:
    """Return an LLM service dependency instance."""

    return LLMService()


def _extract_analysis_result(content: str, fallback_code: str) -> AnalysisResult:
    """Parse and validate JSON content returned by the LLM.

    If the model output is not valid JSON, the function degrades safely
    by returning a structured fallback that preserves the original code.
    """

    payload: Optional[dict[str, Any]] = None

    try:
        loaded = json.loads(content)
        if isinstance(loaded, dict):
            payload = loaded
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(content)
        if match:
            try:
                loaded = json.loads(match.group(0))
                if isinstance(loaded, dict):
                    payload = loaded
            except json.JSONDecodeError:
                payload = None

    if payload is None:
        return AnalysisResult(
            root_cause="Unable to parse structured model output.",
            explanation=(
                "The model response was not valid JSON. Review the raw response "
                "or retry the request."
            ),
            fixed_code=fallback_code,
        )

    root_cause = str(payload.get("root_cause", "")).strip()
    explanation = str(payload.get("explanation", "")).strip()
    fixed_code = str(payload.get("fixed_code", "")).strip()

    return AnalysisResult(
        root_cause=root_cause or "Root cause not provided by model.",
        explanation=explanation or "Explanation not provided by model.",
        fixed_code=fixed_code or fallback_code,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_code_error(
    request: AnalyzeRequest,
    stack_parser: StackTraceParser = Depends(get_stack_trace_parser),
    ast_parser: ASTParserService = Depends(get_ast_parser_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> AnalyzeResponse:
    """Analyze code with stack trace context and return a structured fix."""

    parsed_trace = stack_parser.parse(request.stack_trace)

    parsed_context = ast_parser.parse(
        code=request.code,
        error_line=parsed_trace.error_line,
        filename=request.filename,
    )

    user_prompt = build_analysis_user_prompt(
        filename=request.filename,
        stack_trace_context=parsed_trace.to_prompt_string(),
        ast_context=parsed_context.to_prompt_string(),
        raw_code=request.code,
    )

    try:
        llm_response = await llm_service.generate_response(
            messages=[LLMMessage(role="user", content=user_prompt)],
            system_prompt=ANALYZE_SYSTEM_PROMPT,
            temperature=0.1,
        )
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM analysis failed: {exc}",
        ) from exc

    analysis = _extract_analysis_result(llm_response.content, request.code)

    return AnalyzeResponse(
        language=parsed_trace.language,
        error_type=parsed_trace.error_type,
        error_message=parsed_trace.error_message,
        error_location=ErrorLocation(
            filename=parsed_trace.error_filename,
            line=parsed_trace.error_line,
            function=parsed_trace.error_function,
        ),
        analysis=analysis,
        llm_model=llm_response.model,
        parser_backend=ast_parser.backend_name,
    )
