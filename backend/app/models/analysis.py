"""Pydantic models for the analysis API.

These models define the request and response contract for error analysis.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request payload for the error analysis endpoint."""

    code: str = Field(..., min_length=1, description="Source code to analyze.")
    stack_trace: str = Field(..., min_length=1, description="Raw stack trace text.")
    filename: str = Field(..., min_length=1, description="File name for language detection.")


class ErrorLocation(BaseModel):
    """Normalized location metadata for the detected error frame."""

    filename: Optional[str] = Field(default=None)
    line: Optional[int] = Field(default=None)
    function: Optional[str] = Field(default=None)


class AnalysisResult(BaseModel):
    """Structured LLM output returned to the client."""

    root_cause: str = Field(..., description="Concise technical root cause.")
    explanation: str = Field(..., description="Detailed user-facing explanation.")
    fixed_code: str = Field(..., description="Proposed corrected code.")


class AnalyzeResponse(BaseModel):
    """Response payload for successful analysis."""

    language: str = Field(..., description="Detected stack trace language.")
    error_type: str = Field(..., description="Parsed error type.")
    error_message: str = Field(..., description="Parsed error message.")
    error_location: ErrorLocation = Field(..., description="Best-known crash location.")
    analysis: AnalysisResult = Field(..., description="LLM-generated diagnosis and fix.")
    llm_model: str = Field(..., description="Model used for analysis.")
    parser_backend: str = Field(..., description="AST parser backend name.")
