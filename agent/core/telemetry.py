"""Langfuse telemetry helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency at runtime
    from langfuse import Langfuse
except Exception:  # pragma: no cover - we only need a best-effort import
    Langfuse = None

from agent.config import settings


class TelemetryClient:
    """Wrapper around Langfuse for logging traces."""

    def __init__(self) -> None:
        if Langfuse and settings.langfuse_public_key and settings.langfuse_secret_key:
            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        else:  # pragma: no cover - no external service during tests
            self._client = None

    def log_interaction(
        self,
        *,
        user_query: str,
        response: str,
        retrieved_context: list[Dict[str, Any]],
        error: Optional[str] = None,
    ) -> None:
        if not self._client:  # pragma: no cover - nothing to log in tests
            return

        trace = self._client.trace(
            name="codebase-qa",
            input=user_query,
            output=response,
            session_id=settings.langfuse_session_id,
        )
        trace.event(name="retrieved_context", metadata={"chunks": retrieved_context})
        if error:
            trace.event(name="error", metadata={"message": error})


def get_telemetry_client() -> TelemetryClient:
    return TelemetryClient()
