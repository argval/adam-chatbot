import logging

from dotenv import load_dotenv

from .orchestrator import AgentOrchestrator


def _print_sources(sources) -> None:
    if not sources:
        return
    print("Sources:")
    for idx, item in enumerate(sources, start=1):
        metadata = item.get("metadata") or {}
        source = metadata.get("source_name") or metadata.get("source") or "unknown source"
        score = item.get("score")
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        print(f"  {idx}. {source} (score: {score_text})")


def _print_verification(message: str) -> None:
    if message:
        print(message)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    agent = AgentOrchestrator()
    print("🤖 Agent orchestrator ready. Ask a question or type 'help' / 'exit'.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in {"exit", "quit"}:
            print("Goodbye!")
            break
        if lowered in {"help", "commands"}:
            print(
                "Commands:\n"
                "  - Mention 'ingest' or 'refresh data' to rebuild the knowledge base.\n"
                "  - Mention 'embed' or 'refresh vectors' to regenerate embeddings only.\n"
                "  - Mention 'show sources' or 'retrieve' for context without an answer.\n"
                "  - Any other question triggers retrieval + answer generation.\n"
            )
            continue

        response = agent.handle(user_input)
        print(f"Assistant: {response.get('message', 'No message generated.')}\n")
        _print_verification(response.get("verification"))
        _print_sources(response.get("context") or [])
        print()


if __name__ == "__main__":
    main()
