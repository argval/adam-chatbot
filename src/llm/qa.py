import logging
from typing import Dict, List, Sequence, Tuple

from groq import BadRequestError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.vector_store.query import query_vector_store

from .groq_client import DEFAULT_GROQ_MODEL, get_groq_llm

logger = logging.getLogger(__name__)

MAX_CHARS_PER_CHUNK = 1500

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a friendly, knowledgeable assistant grounded in the provided context. "
                "Answer conversationally, cite evidence as [Source #] that matches the context labels, "
                "and if the context lacks enough information, admit it and suggest a follow-up question."
            ),
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            (
                "Use the following context snippets to answer the latest question.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Provide a concise, helpful answer that references the relevant sources."
            ),
        ),
    ]
)


def _format_context(results: List[Dict], max_chars: int = MAX_CHARS_PER_CHUNK) -> str:
    formatted_chunks: List[str] = []
    for idx, item in enumerate(results, start=1):
        metadata = item.get("metadata", {}) or {}
        source = metadata.get("source", "unknown source")
        content = (item.get("content") or "").strip()

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


def _history_to_messages(
    chat_history: Sequence[Tuple[str, str]] | None,
) -> List[BaseMessage]:
    if not chat_history:
        return []

    messages: List[BaseMessage] = []
    for role, content in chat_history:
        if not content:
            continue

        role_normalized = role.lower()
        if role_normalized in {"user", "human"}:
            messages.append(HumanMessage(content=content))
        elif role_normalized in {"assistant", "ai", "bot"}:
            messages.append(AIMessage(content=content))
        else:
            raise ValueError(f"Unsupported chat history role: {role}")

    return messages


def answer_question(
    question: str,
    vector_store_path: str = "data/vector_store",
    n_results: int = 5,
    model_name: str = DEFAULT_GROQ_MODEL,
    chat_history: Sequence[Tuple[str, str]] | None = None,
) -> Dict[str, object]:
    """
    Generate an answer using Groq's LLM with retrieved vector-store context.

    Args:
        question: User's question.
        vector_store_path: Path to the Chroma DB directory.
        n_results: Number of context chunks to retrieve.
        model_name: Groq model identifier to use.
        chat_history: Sequence of (role, message) tuples from the ongoing conversation.

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
            "answer": "I couldn't find enough information in the knowledge base to answer that yet.",
            "sources": [],
        }

    context = _format_context(results)
    llm = get_groq_llm(model_name=model_name or DEFAULT_GROQ_MODEL)
    chain = QA_PROMPT | llm | StrOutputParser()
    try:
        answer = chain.invoke(
            {
                "question": question,
                "context": context,
                "chat_history": _history_to_messages(chat_history),
            }
        ).strip()
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
