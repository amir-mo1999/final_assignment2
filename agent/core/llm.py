"""Utility helpers for creating LLM clients."""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from agent.config import settings


class MissingOpenAIKey(RuntimeError):
    """Raised when the OpenAI API key is missing."""


def _get_api_key() -> str:
    if settings.openai_api_key is None:
        raise MissingOpenAIKey(
            "OPENAI_API_KEY is required to run the agent. Set it in the environment."
        )
    return settings.openai_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return a cached ChatOpenAI instance."""

    return ChatOpenAI(
        model=settings.openai_model,
        temperature=0.0,
        api_key=_get_api_key(),
    )
