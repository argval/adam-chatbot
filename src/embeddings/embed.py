import json
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = os.getenv("EMBED_MODEL_NAME", DEFAULT_MODEL_NAME)
    collection_name: str = os.getenv("EMBED_COLLECTION_NAME", "documents")
    id_prefix: str = os.getenv("EMBED_ID_PREFIX", "chunk_")
    reset_collection: bool = _env_flag("EMBED_RESET_COLLECTION", False)


def _load_chunks(processed_file: str) -> Sequence[dict]:
    processed_path = Path(processed_file)
    if not processed_path.exists():
        msg = f"Processed file not found at {processed_file}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    with processed_path.open("r", encoding="utf-8") as f:
        docs_data = json.load(f)

    if not docs_data:
        logger.warning("No documents found in processed file.")

    return docs_data


def embed_and_store(
    processed_file: str,
    vector_store_path: str,
    *,
    config: Optional[EmbeddingConfig] = None,
    model_name: Optional[str] = None,
    collection_name: Optional[str] = None,
    reset: Optional[bool] = None,
) -> None:
    """Embed processed chunks and persist them to ChromaDB."""

    cfg = config or EmbeddingConfig()
    if model_name is not None:
        cfg = replace(cfg, model_name=model_name)
    if collection_name is not None:
        cfg = replace(cfg, collection_name=collection_name)
    if reset is not None:
        cfg = replace(cfg, reset_collection=reset)

    docs_data = _load_chunks(processed_file)
    if not docs_data:
        logger.info("No embeddings generated because the processed file is empty.")
        return

    documents = [doc["page_content"] for doc in docs_data]
    metadatas = [doc["metadata"] for doc in docs_data]
    ids = [f"{cfg.id_prefix}{i}" for i in range(len(documents))]

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cfg.model_name
    )

    client = chromadb.PersistentClient(path=vector_store_path)

    if cfg.reset_collection:
        try:
            client.delete_collection(name=cfg.collection_name)
            logger.info("Deleted existing collection '%s' before re-ingesting.", cfg.collection_name)
        except Exception as exc:
            logger.warning("Unable to delete collection '%s': %s", cfg.collection_name, exc)

    collection = client.get_or_create_collection(
        name=cfg.collection_name,
        embedding_function=embedding_function,
    )

    existing_ids = set()
    try:
        snapshot = collection.get(include=["ids"])
        existing_ids.update(snapshot.get("ids", []))
    except Exception as exc:
        logger.debug("Failed to fetch existing IDs: %s", exc)

    stale_ids = list(existing_ids - set(ids))
    if stale_ids:
        logger.info("Removing %s stale vectors from collection '%s'.", len(stale_ids), cfg.collection_name)
        try:
            collection.delete(ids=stale_ids)
        except Exception as exc:
            logger.warning("Failed to delete stale IDs: %s", exc)

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    logger.info(
        "Stored %s chunks in vector store '%s' at %s",
        len(documents),
        cfg.collection_name,
        vector_store_path,
    )


if __name__ == "__main__":
    embed_and_store("data/processed/documents.json", "data/vector_store")
