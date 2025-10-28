import logging
from typing import Dict, List, Sequence, Tuple

from .qa import answer_question

logger = logging.getLogger(__name__)

ChatTurn = Tuple[str, str]


class RAGChatbot:
    """
    Lightweight conversational agent that combines retrieval with Groq's LLM.
    Maintains chat history so answers feel contextual and grounded.
    """

    def __init__(
        self,
        vector_store_path: str = "data/vector_store",
        n_results: int = 5,
        model_name: str | None = None,
    ) -> None:
        self.vector_store_path = vector_store_path
        self.n_results = n_results
        self.model_name = model_name
        self._history: List[ChatTurn] = []

    @property
    def history(self) -> Sequence[ChatTurn]:
        return tuple(self._history)

    def reset(self) -> None:
        self._history.clear()

    def ask(self, question: str) -> Dict[str, object]:
        """
        Answer a user question, updating the conversation history.

        Returns a dictionary containing:
            answer (str): the model's response
            sources (list): metadata for the supporting chunks
            history (tuple): updated conversation history
        """
        response = answer_question(
            question=question,
            vector_store_path=self.vector_store_path,
            n_results=self.n_results,
            model_name=self.model_name,
            chat_history=self._history,
        )

        self._history.extend(
            [
                ("user", question),
                ("assistant", response["answer"]),
            ]
        )

        response["history"] = self.history
        return response


def _format_sources(sources: List[Dict[str, object]]) -> str:
    if not sources:
        return "  (no supporting sources found)"

    lines: List[str] = []
    for idx, item in enumerate(sources, start=1):
        source = item.get("source") or "unknown source"
        score = item.get("score")
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        lines.append(f"  {idx}. {source} (score: {score_text})")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    chatbot = RAGChatbot()

    print("    Ask me anything about your indexed documents.")
    print("    Type 'exit' or 'reset' to quit or clear the conversation.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        lowered = question.lower()
        if lowered in {"exit", "quit"}:
            print("Goodbye!")
            break
        if lowered == "reset":
            chatbot.reset()
            print("Conversation history cleared.")
            continue

        try:
            response = chatbot.ask(question)
        except RuntimeError as exc:
            logger.error("Chatbot failed to answer: %s", exc)
            print("Assistant: Sorry, I ran into an issue. Please try again.")
            continue

        print(f"Assistant: {response['answer']}\n")
        print("Sources:")
        print(_format_sources(response.get("sources", [])))
        print()


if __name__ == "__main__":
    main()
