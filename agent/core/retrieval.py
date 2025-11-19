"""Vector search helpers backed by Postgres + pgvector."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from psycopg2.extras import RealDictCursor

from agent.core.db import get_connection
from langchain_openai import OpenAIEmbeddings
from agent.config import settings


def get_embeddings() -> OpenAIEmbeddings:
    """Return a cached OpenAI embeddings client."""

    return OpenAIEmbeddings(
        model=settings.embeddings_model,
        api_key=settings.openai_api_key,
    )


@dataclass
class PreprocessedQuery:
    original: str
    cleaned: str
    file_filters: List[str]


@dataclass
class RetrievalResult:
    chunks: List[dict]
    processed_query: str
    error: str | None = None


_FILE_MENTION_PATTERN = re.compile(r"[\w./-]+\.py")
_NOISE_PHRASES = [
    "please",
    "could you",
    "would you",
    "can you",
    "tell me",
    "explain",
]


def preprocess_query(query: str) -> PreprocessedQuery:
    """Normalize input text and extract metadata filters."""

    lowered = query.lower().strip()
    cleaned = lowered
    for phrase in _NOISE_PHRASES:
        cleaned = cleaned.replace(phrase, "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    file_matches = [match.group(0) for match in _FILE_MENTION_PATTERN.finditer(query)]
    return PreprocessedQuery(original=query, cleaned=cleaned, file_filters=file_matches)


def _format_file_filter_clause(file_filters: Iterable[str]) -> tuple[str, list]:
    filters = list({path.strip() for path in file_filters if path.strip()})
    if not filters:
        return "", []

    conditions: list[str] = []
    params: list[str] = []
    for file_value in filters:
        if "/" in file_value:
            conditions.append("file_path ILIKE %s")
            params.append(f"%{file_value}%")
        else:
            conditions.append("file_name ILIKE %s")
            params.append(f"%{file_value}%")

    clause = " AND (" + " OR ".join(conditions) + ")"
    return clause, params


def similarity_search(query: str, limit: int = 5) -> RetrievalResult:
    """Execute a similarity search over the code_embeddings table."""

    processed = preprocess_query(query)
    if not processed.cleaned:
        return RetrievalResult(
            chunks=[],
            processed_query=processed.cleaned,
            error="I could not understand that query.",
        )

    # Build query embedding
    try:
        embedding = get_embeddings().embed_query(processed.cleaned)
    except Exception as exc:  # pragma: no cover - depends on OpenAI
        return RetrievalResult(
            chunks=[],
            processed_query=processed.cleaned,
            error=str(exc),
        )

    # Build filter clause (for file filters etc.)
    clause, params = _format_file_filter_clause(processed.file_filters)
    # Ensure clause is a string and includes its own leading whitespace if present
    clause = clause or ""

    sql = f"""
        SELECT
            file_path,
            file_name,
            file_extension,
            chunk_index,
            total_chunks,
            token_count,
            content
        FROM code_embeddings
        WHERE 1=1
        {clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """

    chunks: List[dict] = []

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (*params, embedding, limit))
            rows = cur.fetchall()

    for row in rows:
        chunks.append(
            {
                "file_path": row["file_path"],
                "file_name": row["file_name"],
                "file_extension": row["file_extension"],
                "chunk_index": row["chunk_index"],
                "total_chunks": row["total_chunks"],
                "token_count": row["token_count"],
                "content": row["content"],
            }
        )

    return RetrievalResult(chunks=chunks, processed_query=processed.cleaned)
