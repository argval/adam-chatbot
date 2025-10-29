"""Compatibility wrapper for the legacy chatbot entry point.

The project now uses the agent orchestrator CLI defined in
``src.agent.agent_cli``. This module simply forwards ``main`` so existing
scripts or instructions that import ``src.llm.chatbot`` continue to work.
"""

from src.agent.agent_cli import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
