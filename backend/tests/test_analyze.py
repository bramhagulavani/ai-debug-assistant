"""Integration-style smoke test for the analyze endpoint pipeline."""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

load_dotenv()

from backend.app.api.analyze import AnalyzeRequest, analyze_error


async def _run_analyze_test() -> None:
    """Execute the analyze endpoint with a deterministic IndexError example."""

    code = (
        "class UserManager:\n"
        "    def get_user(self, users, user_id):\n"
        "        return users[user_id]['name']\n\n"
        "manager = UserManager()\n"
        "print(manager.get_user([], 0))\n"
    )

    stack_trace = (
        "Traceback (most recent call last):\n"
        "  File \"main.py\", line 7, in <module>\n"
        "    print(manager.get_user([], 0))\n"
        "  File \"main.py\", line 5, in get_user\n"
        "    return users[user_id]['name']\n"
        "IndexError: list index out of range\n"
    )

    request = AnalyzeRequest(code=code, stack_trace=stack_trace, filename="main.py")
    response = await analyze_error(request)

    print(f"Success: {response.success}")
    print(f"Language: {response.language}")
    print(f"Error type: {response.error_type}")
    print(f"Error message: {response.error_message}")
    print(f"Error line: {response.error_line}")
    print(f"Error fn: {response.error_function}")
    print(f"Duration: {response.duration_ms} ms")
    print("AI RESPONSE:")
    print(response.ai_response)


def test_analyze() -> None:
    """Run the asynchronous analyze test body in a standard test function."""

    asyncio.run(_run_analyze_test())
