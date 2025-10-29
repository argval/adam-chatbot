from typing import Sequence

from .retriever import VectorStoreRetriever, DEFAULT_MODEL_NAME


def query_vector_store(
    query: str,
    vector_store_path: str,
    n_results: int = 5,
    model_name: str = DEFAULT_MODEL_NAME,
    collection_name: str = "documents",
) -> Sequence[dict]:
    retriever = VectorStoreRetriever(
        vector_store_path=vector_store_path,
        model_name=model_name,
        collection_name=collection_name,
    )
    return retriever.query(query=query, n_results=n_results)


if __name__ == "__main__":
    import logging

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
