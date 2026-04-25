"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings sourced from environment variables."""

    openai_api_key: str
    openai_base_url: str
    openai_model: str
    database_url: str
    pinecone_api_key: str
    debug: bool
    jwt_secret_key: str
    jwt_algorithm: str
    access_token_expire_minutes: int
    pinecone_index_name: str = "ai-debug-bugs"

    @classmethod
    def from_env(cls) -> Settings:
        """Construct settings from environment variables and validate required values."""
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "ai-debug-bugs"),

        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")

        debug_raw = os.getenv("DEBUG", "False").strip().lower()
        debug_value = debug_raw in {"1", "true", "yes", "on"}

        return cls(
            openai_api_key=openai_api_key,
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            database_url=os.getenv(
                "DATABASE_URL",
                "postgresql://postgres:postgres@localhost:5432/aidebug",
            ),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            debug=debug_value,
            jwt_secret_key=os.getenv("JWT_SECRET_KEY", "change-me-in-production"),
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")),
        )


settings = Settings.from_env()
