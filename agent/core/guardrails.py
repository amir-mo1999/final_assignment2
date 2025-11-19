"""Guardrails for restricting the agent to codebase QA tasks."""

from __future__ import annotations

import re

from agent.config import settings


class GuardrailViolation(RuntimeError):
    """Raised when the user request falls outside the supported scope."""


_FILE_PATTERN = re.compile(r"[\w./-]+\.py")
_CODE_KEYWORDS = {
    "code",
    "module",
    "function",
    "class",
    "endpoint",
    "api",
    "project",
    "repository",
    "file",
    "auth",
    "authentication",
    "authorization",
    "database",
    "service",
    "handler",
    "router",
}
_BLOCKED_KEYWORDS = {
    "write",
    "create",
    "generate",
    "build",
    "implement",
    "draft",
    "modify",
    "stack overflow",
    "essay",
    "weather",
}


def ensure_supported_query(query: str) -> None:
    """Raise if the query is outside the codebase QA scope."""

    normalized = query.lower().strip()
    if not normalized:
        raise GuardrailViolation(settings.unsupported_query_message)

    if any(keyword in normalized for keyword in _BLOCKED_KEYWORDS):
        raise GuardrailViolation(settings.unsupported_query_message)

    mentions_file = bool(_FILE_PATTERN.search(query))
    mentions_code = any(keyword in normalized for keyword in _CODE_KEYWORDS)

    if not (mentions_file or mentions_code):
        raise GuardrailViolation(settings.unsupported_query_message)


FALLBACK_MESSAGE = settings.unsupported_query_message
