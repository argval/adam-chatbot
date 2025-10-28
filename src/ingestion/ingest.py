import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from document_processing.document_processor_factory import DocumentProcessorFactory

# Set up logging
logging.basicConfig(level=logging.INFO)

def ingest_data(raw_dir: str, processed_dir: str, chunk_size: int = 2200, chunk_overlap: int = 220):
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

    # Ensure processed directory exists
    processed_path.mkdir(parents=True, exist_ok=True)

    factory = DocumentProcessorFactory()

    all_documents = []

    for file_path in raw_path.rglob("*"):
        if file_path.is_file() and factory.get_processor(str(file_path)):
            processor = factory.get_processor(str(file_path))
            if processor:
                logging.info(f"Processing {file_path}")
                result = processor.process(str(file_path), chunk_size, chunk_overlap)
                chunks = result.get("chunks", [])
                all_documents.extend(chunks)

    # Save all documents to a JSON file in processed directory
    output_file = processed_path / "documents.json"
    with open(output_file, "w") as f:
        # Convert documents to dict for JSON serialization
        docs_data = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in all_documents
        ]
        json.dump(docs_data, f, indent=4)

    logging.info(f"Processed {len(all_documents)} documents. Saved to {output_file}")

if __name__ == "__main__":
    # Assuming script is run from project root
    ingest_data("data/raw", "data/processed")
