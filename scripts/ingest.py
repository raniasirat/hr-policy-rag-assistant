"""Ingest HR documents from data/raw/ into the ChromaDB index.

Usage:
    python scripts/ingest.py            # add new / changed documents
    python scripts/ingest.py --reset    # wipe the index first, then re-ingest everything
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigurationError, configure_logging, load_settings  # noqa: E402
from src.document_loader import SUPPORTED_EXTENSIONS, load_directory  # noqa: E402
from src.document_processor import process_documents, save_processed_chunks  # noqa: E402
from src.embeddings import EmbeddingServiceError, get_embeddings, verify_embedding_service  # noqa: E402
from src.vector_store import VectorStoreManager  # noqa: E402

logger = logging.getLogger("ingest")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index HR PDF/DOCX/CSV files into ChromaDB.")
    parser.add_argument("--reset", action="store_true", help="delete the existing index before ingesting")
    parser.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation with --reset")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = load_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level)

    raw_dir = settings.raw_data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Scanning %s", raw_dir)

    result = load_directory(raw_dir)
    if not result.documents:
        print(
            f"\nNo documents to index in {raw_dir}.\n"
            f"Copy PDF or DOCX files there ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) and run this script again."
        )
        for failed in result.failed_files:
            print(f"  skipped: {failed.path.name} - {failed.reason}")
        return 0

    chunks = process_documents(result.documents, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        print("The documents contained no usable text after cleaning.")
        return 0
    processed_path = save_processed_chunks(chunks, settings.processed_data_dir / "chunks.jsonl")

    embeddings = get_embeddings(settings)
    try:
        dimension = verify_embedding_service(embeddings, settings.embedding_model)
    except EmbeddingServiceError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    logger.info("Embedding model '%s' is reachable (dimension %d)", settings.embedding_model, dimension)

    store = VectorStoreManager(settings.chroma_db_path, embeddings, settings.collection_name)
    if args.reset:
        if not args.yes and input(f"Delete the existing index in {settings.chroma_db_path}? [y/N] ").strip().lower() != "y":
            print("Aborted. Nothing was changed.")
            return 1
        store.reset()

    try:
        report = store.add_documents(chunks)
    except Exception as exc:  # embedding failures surface here
        logger.exception("Indexing failed")
        print(f"\nIndexing failed: {exc}", file=sys.stderr)
        return 1

    print("\n=== Ingestion summary ===")
    print(f"Documents loaded     : {len(result.loaded_files)}")
    print(f"Files skipped        : {len(result.failed_files)}")
    for failed in result.failed_files:
        print(f"    - {failed.path.name}: {failed.reason}")
    print(f"Chunks created       : {len(chunks)}")
    print(f"New chunks indexed   : {report.added}")
    print(f"Duplicates skipped   : {report.skipped_duplicates}")
    print(f"Outdated removed     : {report.removed_stale}")
    print(f"Total chunks in index: {store.count()}")
    print(f"Chunks saved to      : {processed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
