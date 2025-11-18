"""CLI entry point for running the ingestion pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from scripts.ingestion.ingestion import ingest_python_repository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a Python repository into pgvector")
    parser.add_argument(
        "--repo-path",
        type=Path,
        required=True,
        help="Path to the Python repository that should be ingested",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    stats = ingest_python_repository(args.repo_path)
    print("Ingestion complete:")
    print(f"  Files processed: {stats['files_processed']}")
    print(f"  Chunks inserted: {stats['chunks_inserted']}")
    print(f"  Chunks skipped: {stats['chunks_skipped']}")


if __name__ == "__main__":
    main()
