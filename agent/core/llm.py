"""Utility helpers for creating LLM clients."""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from agent.config import settings


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return a cached ChatOpenAI instance."""

    return ChatOpenAI(
        model=settings.openai_model,
        temperature=0.0,
        api_key=settings.openai_api_key,
    )
