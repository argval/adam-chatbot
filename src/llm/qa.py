import logging
from typing import Dict, List

from groq import BadRequestError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.vector_store.query import query_vector_store

from .groq_client import DEFAULT_GROQ_MODEL, get_groq_llm

logger = logging.getLogger(__name__)

MAX_CHARS_PER_CHUNK = 1500

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a helpful research assistant. "
                "Use the provided context snippets to answer the user's question. "
                "If the answer cannot be determined from the context, explicitly state that."
            ),
        ),
        (
            "human",
            "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:",
        ),
    ]
)


def _format_context(results: List[Dict], max_chars: int = MAX_CHARS_PER_CHUNK) -> str:
    formatted_chunks: List[str] = []
    for idx, item in enumerate(results, start=1):
        metadata = item.get("metadata", {}) or {}
        source = metadata.get("source", "unknown source")
        content = item.get("content", "").strip()

        if not content:
            continue

        if len(content) > max_chars:
            content = content[: max_chars - 3].rstrip() + "..."

        formatted_chunks.append(f"[Source {idx}] ({source})\n{content}")

    return "\n\n".join(formatted_chunks)


def _summarize_sources(results: List[Dict]) -> List[Dict[str, object]]:
    return [
        {
            "source": (item.get("metadata") or {}).get("source"),
            "score": item.get("score"),
        }
        for item in results
    ]


def answer_question(
    question: str,
    vector_store_path: str = "data/vector_store",
    n_results: int = 5,
    model_name: str = DEFAULT_GROQ_MODEL,
) -> Dict[str, object]:
    """
    Generate an answer using Groq's Llama3 model with retrieved vector-store context.

    Args:
        question: User's question.
        vector_store_path: Path to the Chroma DB directory.
        n_results: Number of context chunks to retrieve.
        model_name: Groq model identifier to use.

    Returns:
        Dict containing the `answer` text and the `sources` metadata list.
    """
    if not question:
        raise ValueError("Question must be a non-empty string.")

    results = query_vector_store(
        query=question,
        vector_store_path=vector_store_path,
        n_results=n_results,
    )

    if not results:
        logger.info("No retrieval results for question: %s", question)
        return {
            "answer": "I could not find enough information in the knowledge base to answer that.",
            "sources": [],
        }

    context = _format_context(results)
    llm = get_groq_llm(model_name=model_name)
    chain = QA_PROMPT | llm | StrOutputParser()
    try:
        answer = chain.invoke({"question": question, "context": context}).strip()
    except BadRequestError as exc:
        raise RuntimeError(
            "Groq rejected the request. Verify GROQ_MODEL_NAME points to a supported model."
        ) from exc

    return {
        "answer": answer,
        "sources": _summarize_sources(results),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        response = answer_question("Summarize the key idea in the documents.")
    except RuntimeError as exc:
        logger.error("LLM pipeline failed: %s", exc)
    else:
        logger.info("Answer: %s", response["answer"])
        logger.info("Sources: %s", response["sources"])
