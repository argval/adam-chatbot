import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class VerificationConfig:
    """
    Configuration for verification and self-review heuristics.
    """

    min_token_length: int = int(os.getenv("AGENT_VERIFICATION_MIN_TOKEN", "4"))
    min_sentence_length: int = int(os.getenv("AGENT_VERIFICATION_MIN_SENTENCE", "24"))
    stopwords: set[str] = field(
        default_factory=lambda: {
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
            "show",
            "tell",
            "give",
            "experience",
        }
    )


class VerificationHelper:
    def __init__(self, config: Optional[VerificationConfig] = None) -> None:
        self.config = config or VerificationConfig()

    # Basic verification -----------------------------------------------------
    def basic(self, question: str, sources: List[Dict[str, object]]) -> str:
        if not sources:
            return "⚠️ Verification: No supporting context retrieved."

        keywords = self._question_keywords(question)
        if not keywords:
            return "✅ Verification: Retrieved context covers the main query terms."

        covered = set()
        for item in sources:
            content = (item.get("content") or "").lower()
            for token in keywords:
                if token in content:
                    covered.add(token)
        missing = sorted(keywords - covered)
        if missing:
            return (
                "⚠️ Verification: Some query terms were not found in the retrieved context "
                f"({', '.join(missing)})."
            )
        return "✅ Verification: Retrieved context covers the main query terms."

    # Self review -----------------------------------------------------------
    def self_check(self, answer: str, sources: List[Dict[str, object]]) -> str:
        context_blob = " ".join(
            (item.get("content") or "").lower() for item in sources if item
        ).strip()
        if not context_blob:
            return "⚠️ Self-check: No context available to verify the answer."

        issues: List[str] = []
        for sentence in self._split_sentences(answer):
            if len(sentence) < self.config.min_sentence_length:
                continue
            tokens = [
                token
                for token in self._sentence_tokens(sentence)
                if token not in self.config.stopwords
            ]
            if not tokens:
                continue
            if any(token in context_blob for token in tokens):
                continue
            preview = sentence[:80] + ("…" if len(sentence) > 80 else "")
            issues.append(f"• \"{preview}\"")

        if issues:
            return (
                "⚠️ Self-check: These statements were not confirmed in the retrieved passages:\n"
                + "\n".join(issues)
            )
        return "✅ Self-check: Answer statements are supported by retrieved context."

    # Helpers ----------------------------------------------------------------
    def _question_keywords(self, question: str) -> set[str]:
        tokens = {
            token.lower()
            for token in question.split()
            if len(token) >= self.config.min_token_length and token.isalpha()
        }
        return tokens - self.config.stopwords

    def _split_sentences(self, text: str) -> List[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ]

    def _sentence_tokens(self, sentence: str) -> List[str]:
        return [
            token.lower()
            for token in re.findall(r"[A-Za-z]{%d,}" % self.config.min_token_length, sentence)
        ]
