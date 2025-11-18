"""Shared embedding client for ingestion and retrieval."""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from agent.config import settings


class MissingOpenAIKey(RuntimeError):
    """Raised when OpenAI credentials are not configured."""


def _get_api_key() -> str:
    if settings.openai_api_key is None:
        raise MissingOpenAIKey(
            "OPENAI_API_KEY is required to generate embeddings. Set it in the environment."
        )
    return settings.openai_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """Return a cached OpenAI embeddings client."""

    return OpenAIEmbeddings(
        model=settings.embeddings_model,
        api_key=_get_api_key(),
    )
