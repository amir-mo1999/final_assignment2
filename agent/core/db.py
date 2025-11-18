"""Database helpers for connecting to Postgres/pgvector."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector

from agent.config import settings


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """Yield a Postgres connection with pgvector registered."""

    conn = psycopg.connect(settings.postgres_dsn, autocommit=True)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()
