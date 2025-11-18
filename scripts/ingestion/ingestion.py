"""Python-specific ingestion pipeline."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import tiktoken
from psycopg import Connection
from psycopg.rows import dict_row

from agent.core.db import get_connection
from scripts.ingestion.embeddings import get_embeddings

_ENCODER = tiktoken.get_encoding("cl100k_base")
_EXCLUDED_DIRS = {".git", "__pycache__", ".venv", "venv"}


@dataclass
class CodeChunk:
    file_path: str
    file_name: str
    file_extension: str
    chunk_index: int
    total_chunks: int
    content: str
    language: str
    token_count: int
    content_hash: str


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _node_source(content: str, node: ast.AST) -> str:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return content
    lines = content.splitlines()
    snippet = lines[node.lineno - 1: node.end_lineno]
    return "\n".join(snippet).strip()


def _chunk_by_functions(tree: ast.AST, content: str) -> List[str]:
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]  # type: ignore[attr-defined]
    return [_node_source(content, fn) for fn in functions if hasattr(fn, "lineno")]


def _chunk_by_classes(tree: ast.AST, content: str) -> List[str]:
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]  # type: ignore[attr-defined]
    return [_node_source(content, cls) for cls in classes if hasattr(cls, "lineno")]


def chunk_python_file(path: Path, repo_root: Path) -> List[CodeChunk]:
    """Chunk a Python file by AST function, class, or file fallback."""

    raw_content = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(raw_content)
    except SyntaxError:
        tree = ast.parse("pass")
        tree.body = []  # type: ignore[attr-defined]

    chunks: List[str] = _chunk_by_functions(tree, raw_content)
    if not chunks:
        chunks = _chunk_by_classes(tree, raw_content)
    if not chunks:
        chunks = [raw_content]

    rel_path = str(path.relative_to(repo_root))
    total_chunks = len(chunks)
    results: List[CodeChunk] = []
    for idx, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        content_hash = _hash_content(f"{rel_path}:{idx}:{chunk}")
        results.append(
            CodeChunk(
                file_path=rel_path,
                file_name=path.name,
                file_extension=path.suffix,
                chunk_index=idx,
                total_chunks=total_chunks,
                content=chunk,
                language="python",
                token_count=_count_tokens(chunk),
                content_hash=content_hash,
            )
        )
    return results


def _iter_python_files(repo_root: Path) -> Iterable[Path]:
    for path in repo_root.rglob("*.py"):
        if any(part in _EXCLUDED_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def _chunk_exists(conn: Connection, content_hash: str) -> bool:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT 1 FROM code_embeddings WHERE content_hash = %s",
            (content_hash,),
        )
        return cur.fetchone() is not None


def _insert_chunk(conn: Connection, chunk: CodeChunk, embedding: List[float]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO code_embeddings (
                file_path,
                file_name,
                file_extension,
                content,
                content_hash,
                language,
                chunk_index,
                total_chunks,
                embedding,
                token_count
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (content_hash) DO NOTHING
            """,
            (
                chunk.file_path,
                chunk.file_name,
                chunk.file_extension,
                chunk.content,
                chunk.content_hash,
                chunk.language,
                chunk.chunk_index,
                chunk.total_chunks,
                embedding,
                chunk.token_count,
            ),
        )


def ingest_python_repository(repo_root: Path) -> dict:
    """Ingest all Python files under the provided repository path."""

    if not repo_root.exists():
        raise FileNotFoundError(f"Repository path {repo_root} does not exist")

    embedding_client = get_embeddings()
    processed_files = 0
    inserted_chunks = 0
    skipped_chunks = 0

    with get_connection() as conn:
        for file_path in _iter_python_files(repo_root):
            processed_files += 1
            chunks = chunk_python_file(file_path, repo_root)
            if not chunks:
                continue
            vectors = embedding_client.embed_documents([chunk.content for chunk in chunks])
            for chunk, vector in zip(chunks, vectors):
                if _chunk_exists(conn, chunk.content_hash):
                    skipped_chunks += 1
                    continue
                _insert_chunk(conn, chunk, vector)
                inserted_chunks += 1

    return {
        "files_processed": processed_files,
        "chunks_inserted": inserted_chunks,
        "chunks_skipped": skipped_chunks,
    }
