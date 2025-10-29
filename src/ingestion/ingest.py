import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.document_processing.document_processor_factory import (
    DocumentProcessorFactory,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    processed_files: int = 0
    skipped_files: int = 0
    error_files: int = 0
    generated_chunks: int = 0
    skipped_paths: List[str] = field(default_factory=list)
    error_paths: List[str] = field(default_factory=list)


def ingest_data(
    raw_dir: str,
    processed_dir: str,
    chunk_size: int = 2200,
    chunk_overlap: int = 220,
) -> IngestionStats:
    """
    Ingests documents from the raw directory, processes them, and saves to the processed directory.

    Args:
        raw_dir (str): Path to the raw data directory.
        processed_dir (str): Path to the processed data directory.
        chunk_size (int): Chunk size for splitting documents.
        chunk_overlap (int): Overlap between chunks.
    """
    raw_path = Path(raw_dir).resolve()
    processed_path = Path(processed_dir).resolve()
    processed_path.mkdir(parents=True, exist_ok=True)

    factory = DocumentProcessorFactory()
    all_documents = []
    stats = IngestionStats()

    logger.info("Starting ingestion from %s", raw_path)

    for file_path in raw_path.rglob("*"):
        if not file_path.is_file():
            continue

        processor = factory.get_processor(str(file_path))
        if not processor:
            stats.skipped_files += 1
            stats.skipped_paths.append(str(file_path))
            logger.debug("No processor registered for %s", file_path)
            continue

        logger.info("Processing %s with %s", file_path, processor.__class__.__name__)
        try:
            result = processor.process(str(file_path), chunk_size, chunk_overlap)
        except Exception as exc:
            stats.error_files += 1
            stats.error_paths.append(str(file_path))
            logger.exception("Failed to process %s: %s", file_path, exc)
            continue

        chunks = result.get("chunks", [])
        all_documents.extend(chunks)
        stats.processed_files += 1
        stats.generated_chunks += len(chunks)

    output_file = processed_path / "documents.json"
    docs_data = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in all_documents
    ]
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(docs_data, f, indent=4)

    logger.info(
        "Ingestion summary: %s processed, %s skipped, %s errored, %s chunks generated.",
        stats.processed_files,
        stats.skipped_files,
        stats.error_files,
        stats.generated_chunks,
    )
    if stats.skipped_files:
        logger.debug("Skipped files: %s", stats.skipped_paths)
    if stats.error_files:
        logger.debug("Files with errors: %s", stats.error_paths)
    logger.info("Saved processed documents to %s", output_file)

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_data("data/raw", "data/processed")
