"""Prompt builders used by the debug analysis endpoint."""

from __future__ import annotations


def build_debug_system_prompt() -> str:
    """Return the system prompt that constrains output format and behavior."""

    return (
        "You are an expert debugging assistant. "
        "Always return exactly four sections with these exact headings:\n"
        "## Root Cause\n"
        "1-2 sentences naming the exact variable and/or line that caused the failure.\n\n"
        "## Explanation\n"
        "2-4 sentences in plain English explaining why the error happens.\n\n"
        "## Fixed Code\n"
        "Provide the corrected full function and include an inline comment on the changed line.\n\n"
        "## Prevention\n"
        "Exactly 1 sentence describing how to avoid this class of error.\n\n"
        "Rules: do not invent code that is not present in the supplied source context, "
        "avoid filler phrases, and be direct."
    )


def build_debug_user_prompt(stack_trace_context: str, ast_context: str, raw_code: str) -> str:
    """Return the user prompt containing structured error and source-code context."""

    return (
        "=== ERROR INFORMATION ===\n"
        f"{stack_trace_context}\n\n"
        "=== CODE STRUCTURE ===\n"
        f"{ast_context}\n\n"
        "=== FULL SOURCE CODE ===\n"
        f"```\n{raw_code}\n```\n\n"
        "Analyse the error and return your response in the four sections specified."
    )
