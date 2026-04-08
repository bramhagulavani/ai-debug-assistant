# AI Debug Assistant

## Architecture

This repository is organized as a three-part product: backend API, frontend web app, and VS Code extension. The current scaffold is intentionally empty so implementation can be added module by module without mixing concerns.

## Project Layout

```text
ai-debug-assistant/
  backend/
    app/
      api/         # route handlers
      services/    # llm_service.py, ast_parser.py, vector_service.py
      models/      # SQLAlchemy + Pydantic schemas
      prompts/     # structured prompt templates
      core/        # config, auth, database session
    tests/
    Dockerfile
    requirements.txt
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
  docker-compose.yml
  README.md
```

## Responsibility Split

- Backend: exposes analysis APIs, manages persistence, and orchestrates parsing and LLM calls.
- Frontend: provides the browser-based dashboard for submitting errors, reviewing diffs, and tracking session history.
- VS Code extension: embeds the assistant in the editor, captures context, and launches the sidebar workflow.
- Docs: records the system design, API contract, and operational notes.

## Notes

- No implementation files have been added yet.
- This README is the only populated file in the scaffold for now.
