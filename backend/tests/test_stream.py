"""Test the WebSocket streaming endpoint.

Starts the FastAPI app in a test client and connects via WebSocket.
Prints each token as it arrives so you can see streaming in action.
"""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from fastapi.testclient import TestClient

from backend.main import app

TEST_CODE = """
import os

class UserManager:
    def get_user(self, users, user_id):
        return users[user_id]['name']

def main():
    mgr = UserManager()
    print(mgr.get_user([], 'john'))
"""

TEST_TRACE = """
Traceback (most recent call last):
  File "main.py", line 10, in main
    print(mgr.get_user([], 'john'))
  File "main.py", line 5, in get_user
    return users[user_id]['name']
IndexError: list index out of range
"""

payload = json.dumps(
    {
        "code": TEST_CODE,
        "stack_trace": TEST_TRACE,
        "filename": "main.py",
    }
)


def test_websocket_stream() -> None:
    """Connect to the WebSocket endpoint and print streamed tokens."""

    client = TestClient(app)

    print("\nConnecting to WebSocket...")
    print("=" * 50)
    print("STREAMING RESPONSE:")
    print("=" * 50)

    full_response = []

    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_text(payload)

        while True:
            message = ws.receive_text()

            # Check if this is a control frame (done or error)
            try:
                frame = json.loads(message)
                if isinstance(frame, dict):
                    if frame.get("type") == "done":
                        print(f"\n{'=' * 50}")
                        print(f"Done. Duration: {frame.get('duration_ms')}ms")
                        break
                    if frame.get("type") == "error":
                        print(f"\nERROR: {frame.get('message')}")
                        break
                    print(message, end="", flush=True)
                    full_response.append(message)
                else:
                    print(message, end="", flush=True)
                    full_response.append(message)
            except json.JSONDecodeError:
                # Plain text token — print immediately without newline
                print(message, end="", flush=True)
                full_response.append(message)

    print(f"\nTotal tokens received: {len(full_response)}")


if __name__ == "__main__":
    test_websocket_stream()