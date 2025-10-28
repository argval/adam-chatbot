import json
import logging
from pathlib import Path
from typing import Sequence

import chromadb
from chromadb.utils import embedding_functions

# Set up logging
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def _load_chunks(processed_file: str) -> Sequence[dict]:
    processed_path = Path(processed_file)
    if not processed_path.exists():
        msg = f"Processed file not found at {processed_file}"
        logging.error(msg)
        raise FileNotFoundError(msg)

    with processed_path.open("r") as f:
        docs_data = json.load(f)

    if not docs_data:
        logging.warning("No documents found in processed file.")

    return docs_data


def embed_and_store(
    processed_file: str,
    vector_store_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    collection_name: str = "documents",
):
    """
    Loads chunks from processed JSON, generates embeddings, and stores in ChromaDB.

    Args:
        processed_file (str): Path to documents.json.
        vector_store_path (str): Path to store the vector database.
    """
    docs_data = _load_chunks(processed_file)

    if not docs_data:
        return

    documents = [doc["page_content"] for doc in docs_data]
    metadatas = [doc["metadata"] for doc in docs_data]
    ids = [f"chunk_{i}" for i in range(len(documents))]

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )

    client = chromadb.PersistentClient(path=vector_store_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    logging.info(
        "Stored %s chunks in vector store at %s",
        len(documents),
        vector_store_path,
    )


if __name__ == "__main__":
    embed_and_store("data/processed/documents.json", "data/vector_store")
