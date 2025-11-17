from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    openai_api_key: str = Field(default="")

    env: str = "dev"

    class Config:
        env_file = ".env"


settings = Settings()
