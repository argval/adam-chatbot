import logging
from typing import List, Optional, Sequence

import chromadb
from chromadb.utils import embedding_functions

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def query_vector_store(
    query: str,
    vector_store_path: str,
    n_results: int = 5,
    model_name: str = DEFAULT_MODEL_NAME,
    collection_name: str = "documents",
) -> Sequence[dict]:
    """
    Queries the vector store for relevant chunks.

    Args:
        query (str): The query string.
        vector_store_path (str): Path to the vector database.
        n_results (int): Number of results to return.

    Returns:
        List of relevant documents.
    """
    if not query:
        raise ValueError("Query must be a non-empty string.")

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    client = chromadb.PersistentClient(path=vector_store_path)

    try:
        collection = client.get_collection(
            name=collection_name, embedding_function=embedding_function
        )
    except chromadb.errors.InvalidCollectionException as exc:
        msg = (
            f"Collection '{collection_name}' not found at '{vector_store_path}'. "
            "Run the embedding pipeline first."
        )
        raise RuntimeError(msg) from exc

    results = collection.query(query_texts=[query], n_results=n_results)

    if not results.get("documents"):
        logging.info("No results returned for query: %s", query)
        return []

    formatted_results: List[dict] = []
    for idx, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][idx]
        score: Optional[float] = None
        if "distances" in results and results["distances"]:
            score = results["distances"][0][idx]
        formatted_results.append(
            {
                "content": document,
                "metadata": metadata,
                "score": score,
            }
        )

    return formatted_results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        results = query_vector_store(
            query="What is the attention mechanism?",
            vector_store_path="data/vector_store",
        )
    except RuntimeError as exc:
        logging.error(exc)
    else:
        for res in results:
            preview = res["content"][:100].replace("\n", " ")
            logging.info("Match: %s... (Score: %s)", preview, res["score"])
