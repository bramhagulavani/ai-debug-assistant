"""Prompt templates for code + stack trace analysis."""

from __future__ import annotations

ANALYZE_SYSTEM_PROMPT = """You are a senior debugging assistant.
You will receive:
1) Parsed stack trace details.
2) Parsed AST/code context.
3) Raw source code.

Your task:
- Identify the most likely root cause.
- Explain the issue clearly and technically.
- Return a corrected version of the code.

Output rules:
- Return valid JSON only.
- Do not wrap JSON in markdown.
- Use exactly these keys: root_cause, explanation, fixed_code.
- Keep root_cause concise (1-2 sentences).
- Keep explanation actionable and specific to the provided code.
- Ensure fixed_code is complete and runnable for the provided snippet.
"""


def build_analysis_user_prompt(
    *,
    filename: str,
    stack_trace_context: str,
    ast_context: str,
    raw_code: str,
) -> str:
    """Build the user prompt payload for the analysis pipeline."""

    return (
        f"[Filename]\n{filename}\n\n"
        f"[Stack Trace Context]\n{stack_trace_context}\n\n"
        f"[AST Context]\n{ast_context}\n\n"
        f"[Raw Source Code]\n{raw_code}\n"
    )
