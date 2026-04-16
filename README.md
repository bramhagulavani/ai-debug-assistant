# AI Debug Assistant

An AI-assisted debugging workspace with three surfaces:
- Backend API and orchestration layer
- Frontend web experience
- VS Code extension integration

## Architecture

This repository is organized as a three-part product: backend API, frontend web app, and VS Code extension.

## Project Layout

```text
ai-debug-assistant/
  backend/
    app/
      api/         # route handlers
      services/    # llm_service.py and service abstractions
      models/      # SQLAlchemy + Pydantic schemas
      prompts/     # structured prompt templates
      core/        # config, auth, database session
    tests/
  frontend/
    src/
      components/ # ErrorInput, DiffViewer, AnalysisPanel, Dashboard
      hooks/      # useDebugSession, useStreamingResponse
      pages/      # Home, Dashboard, History, Settings
      services/   # api.ts - typed API client
  vscode-extension/
    src/extension.ts    # activation, commands, sidebar provider
    src/webview/        # sidebar panel HTML/CSS/JS
    package.json
  docs/                 # architecture diagram, API reference
  README.md
```

## Responsibility Split

- Backend: exposes analysis APIs, manages persistence, and orchestrates parsing and LLM calls.
- Frontend: provides the browser-based dashboard for submitting errors, reviewing diffs, and tracking session history.
- VS Code extension: embeds the assistant in the editor, captures context, and launches the sidebar workflow.
- Docs: records the system design, API contract, and operational notes.

## Notes

- Backend currently includes an OpenAI-backed LLM service at backend/app/services/llm_service.py.
- The LLM service exposes:
  - LLMService.model
  - LLMService.generate_response(...)
  - LLMService.stream_response(...)

## Streaming API

The backend now exposes a WebSocket endpoint for token-by-token streaming at `/ws/analyze`.

Request payload:

```json
{
  "code": "...",
  "stack_trace": "...",
  "filename": "main.py",
  "language": "python"
}
```

Response behavior:

- Plain text frames contain streamed tokens.
- The final frame is JSON: `{"type":"done","duration_ms":1234}`.
- Error frames use JSON: `{"type":"error","message":"..."}`.

## Local Backend Setup

1. Create and activate a virtual environment.
2. Install dependencies:
  - pip install -r requirements.txt
3. Add environment values to a local .env file:
  - OPENAI_API_KEY=your_key
  - OPENAI_BASE_URL=https://api.openai.com/v1 (optional)

## Quick Verification

From the project root, run:

python -c "from dotenv import load_dotenv; load_dotenv(); from backend.app.services.llm_service import LLMService; print(LLMService().model)"

Expected output:
- gpt-4o

## WebSocket Smoke Test

To exercise the streaming endpoint end to end, run:

python -m backend.tests.test_stream

This opens a test client connection to `/ws/analyze`, streams tokens to the console, and prints the final duration once the response completes.
