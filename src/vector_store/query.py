import logging
import re
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

    if not _has_relevant_hits(results, query):
        candidate_results: Optional[dict] = None
        for keyword in _extract_keywords(query):
            logging.debug("Trying lexical fallback for keyword: %s", keyword)
            fallback_results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where_document={"$contains": keyword},
            )
            if not _has_documents(fallback_results):
                continue

            if _has_relevant_hits(fallback_results, query):
                results = fallback_results
                break

            if candidate_results is None:
                candidate_results = fallback_results
        else:
            if candidate_results and not _has_relevant_hits(results, query):
                results = candidate_results

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


def _canonical_keyword(keyword: str) -> str:
    canonical = keyword.lower().strip("\"'")
    canonical = re.sub(r"[’']s$", "", canonical)
    if canonical.endswith("s") and len(canonical) > 3:
        canonical = canonical[:-1]
    return canonical


def _extract_keywords(text: str) -> List[str]:
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z'\\-]+", text)
    stop_words = {
        "what",
        "where",
        "when",
        "which",
        "who",
        "whose",
        "whom",
        "why",
        "how",
        "does",
        "the",
        "that",
        "this",
        "with",
        "from",
        "into",
        "about",
        "for",
        "your",
        "have",
        "has",
        "know",
        "look",
        "like",
    }

    primary: List[str] = []
    secondary: List[str] = []
    seen: set[str] = set()
    seen_canonical: set[str] = set()

    def add_keyword(keyword: str, bucket: List[str]) -> None:
        if not keyword:
            return
        canonical = _canonical_keyword(keyword)
        if canonical in seen_canonical:
            return
        seen.add(keyword.lower())
        seen_canonical.add(canonical)
        bucket.append(keyword)

    for token in raw_tokens:
        token_clean = token.strip("'\"")
        base = token_clean
        if base.endswith(("'s", "’s")):
            base = base[:-2]

        base_lower = base.lower()
        if len(base) < 3 or base_lower in stop_words:
            continue

        if token[0].isupper():
            add_keyword(base, primary)
            continue

        if len(base) >= 5:
            add_keyword(base, secondary)
        elif len(base) >= 4:
            add_keyword(base, secondary)

    return primary + secondary


def _has_relevant_hits(results: dict, query: str) -> bool:
    if not _has_documents(results):
        return False

    keywords = {_canonical_keyword(kw) for kw in _extract_keywords(query)}
    if not keywords:
        return True

    matched: set[str] = set()
    for doc in results["documents"][0]:
        lowered = doc.lower()
        for keyword in keywords:
            if keyword and keyword in lowered:
                matched.add(keyword)

    required = 1 if len(keywords) == 1 else min(len(keywords), 2)
    return len(matched) >= required


def _has_documents(results: dict) -> bool:
    docs = results.get("documents")
    return bool(docs and docs[0])

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
