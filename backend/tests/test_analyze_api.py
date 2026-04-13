"""Integration-style tests for the /api/analyze endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.api.analyze import get_llm_service
from backend.app.main import create_app
from backend.app.services.llm_service import LLMResponse


class _FakeLLMService:
    """Fake async LLM service used to isolate API behavior in tests."""

    async def generate_response(self, *args, **kwargs) -> LLMResponse:  # noqa: ANN002, ANN003
        """Return deterministic JSON content for endpoint validation."""

        return LLMResponse(
            content=(
                '{"root_cause":"IndexError from invalid list access",'
                '"explanation":"The code indexes an empty list; add a bounds check.",'
                '"fixed_code":"def get_user(users, user_id):\\n'
                '    if user_id >= len(users):\\n'
                '        return None\\n'
                '    return users[user_id]"}'
            ),
            model="fake-gpt-4o",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )


def test_analyze_endpoint_returns_structured_payload() -> None:
    """POST /api/analyze should return parsed metadata and analysis JSON."""

    app = create_app()
    app.dependency_overrides[get_llm_service] = _FakeLLMService
    client = TestClient(app)

    payload = {
        "code": "def get_user(users, user_id):\\n    return users[user_id]",
        "stack_trace": (
            "Traceback (most recent call last):\\n"
            "  File \"main.py\", line 10, in main\\n"
            "    result = get_user([], 0)\\n"
            "  File \"main.py\", line 2, in get_user\\n"
            "    return users[user_id]\\n"
            "IndexError: list index out of range"
        ),
        "filename": "main.py",
    }

    response = client.post("/api/analyze", json=payload)

    assert response.status_code == 200
    body = response.json()

    assert body["language"] == "python"
    assert body["error_type"] == "IndexError"
    assert body["error_location"]["filename"] == "main.py"
    assert body["error_location"]["line"] == 2
    assert body["analysis"]["root_cause"] == "IndexError from invalid list access"
    assert "bounds check" in body["analysis"]["explanation"]
    assert "if user_id >= len(users):" in body["analysis"]["fixed_code"]
    assert body["llm_model"] == "fake-gpt-4o"

    app.dependency_overrides.clear()
