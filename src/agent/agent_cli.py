import logging

from dotenv import load_dotenv


from .orchestrator import AgentOrchestrator

COMMAND_HELP = """
Commands available:
  • ingest / refresh data      → Re-run ingestion and embeddings
  • embed / refresh vectors    → Rebuild embeddings from processed docs
  • status                     → Show last ingestion/embed timestamps and counts
  • show sources <query>       → Retrieve passages without answering
  • help                       → Display this message
  • exit / quit                → Leave the session
""".strip()


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
    print("🤖 Agent orchestrator ready. Type 'help' to see available commands.")

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
            print(COMMAND_HELP)
            continue

        response = agent.handle(user_input)
        message = response.get("message", "No message generated.")
        print(f"Assistant: {message}\n")

        verification = response.get("verification")
        _print_verification(verification)
        sources = response.get("context") or []
        _print_sources(sources)

        if not sources or (verification and verification.startswith("⚠️")):
            print(
                "Tip: try `show sources <query>` or re-run `ingest` if the dataset changed."
            )

        print()


if __name__ == "__main__":
    main()
