from pydantic_settings import BaseSettings
from pydantic import SecretStr
from typing import Optional


class Settings(BaseSettings):
    openai_api_key: Optional[SecretStr] = None

    env: str = "dev"

    class Config:
        env_file = ".env"


settings = Settings()

if settings.openai_api_key is None:
    raise RuntimeError(
        "OPENAI_API_KEY is required. Set it in the environment or .env file."
    )
