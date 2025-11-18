"""Database helpers for connecting to Postgres/pgvector."""

from __future__ import annotations

from contextlib import contextmanager

import psycopg2
from pgvector.psycopg2 import register_vector
from agent.config import settings


@contextmanager
def get_connection():
    """Yield a Postgres connection using psycopg2 with pgvector adapter."""
    conn = psycopg2.connect(settings.postgres_dsn)
    register_vector(conn)
    try:
        yield conn
    finally:
        conn.close()
