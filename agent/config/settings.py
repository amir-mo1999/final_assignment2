from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Runtime configuration for the codebase QA agent."""

    env: str = "dev"

    # OpenAI
    openai_api_key: Optional[SecretStr] = None
    openai_model: str = "gpt-4o-mini"
    embeddings_model: str = "text-embedding-3-small"

    # Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_user: str = "assistant_user"
    postgres_password: str = "assistant_pass"
    postgres_db: str = "coding_assistant"

    # Langfuse
    langfuse_base_url: Optional[str] = "http://localhost:3000"
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_session_id: str = "local-cli"

    # Guardrails
    unsupported_query_message: str = (
        "I can only answer questions about the ingested Python codebase. "
        "Please rephrase your request with details about the project files or architecture."
    )

    class Config:
        env_file = ".env"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"  # nosec: B108
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
