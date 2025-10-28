import os
from functools import lru_cache

from langchain_groq import ChatGroq

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")


@lru_cache(maxsize=4)
def get_groq_llm(
    model_name: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.1,
    max_tokens: int | None = None,
) -> ChatGroq:
    """
    Return a cached Groq chat model instance.

    Args:
        model_name: Groq model identifier to use.
        temperature: Sampling temperature for the response.
        max_tokens: Optional cap on generated tokens.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export the key or add it to your environment."
        )

    # ChatGroq reads the environment variable, but we pass it explicitly to be safe.
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
