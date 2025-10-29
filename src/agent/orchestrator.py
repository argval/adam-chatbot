import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
import re
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.embeddings.embed import embed_and_store
from src.ingestion.ingest import ingest_data
from src.llm.qa import answer_question
from src.vector_store.query import query_vector_store

load_dotenv()

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    intent: str
    context: List[Dict[str, object]]
    answer: str
    message: str
    actions: List[str]
    verification: str


@dataclass
class AgentOrchestrator:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    vector_store_path: str = "data/vector_store"
    k: int = 5
    cross_encoder_model: Optional[str] = None
    history: List[tuple[str, str]] = field(default_factory=list)

    _STOPWORDS = {
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

    def __post_init__(self) -> None:
        self.cross_encoder = self._load_cross_encoder(self.cross_encoder_model)
        self.status_file = Path("data/status.json")
        self.logs_file = Path("data/logs/conversations.jsonl")
        self.graph = self._build_graph()

    # Public API -----------------------------------------------------------------
    def handle(self, question: str) -> AgentState:
        if not question:
            return {
                "message": "Please provide a non-empty question.",
                "actions": ["noop"],
            }

        initial_state: AgentState = {
            "question": question,
            "actions": [],
        }
        response: AgentState = self.graph.invoke(initial_state)

        if "answer" in response:
            self.history.extend(
                [
                    ("user", question),
                    ("assistant", response["answer"]),
                ]
            )

        return response

    # Graph construction --------------------------------------------------------
    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("plan", self._plan_node)
        graph.add_node("ingest", self._ingest_node)
        graph.add_node("embed", self._embed_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("answer", self._answer_node)
        graph.add_node("status", self._status_node)
        graph.add_node("verify", self._verify_node)
        graph.add_node("finalize", self._finalize_node)

        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", self._route_from_plan)
        graph.add_edge("ingest", "finalize")
        graph.add_edge("embed", "finalize")
        graph.add_edge("retrieve", "finalize")
        graph.add_edge("status", "finalize")
        graph.add_edge("answer", "verify")
        graph.add_edge("verify", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    # Graph nodes ----------------------------------------------------------------
    def _plan_node(self, state: AgentState) -> AgentState:
        question = state["question"].lower()
        if any(key in question for key in ("ingest", "re-index", "refresh data")):
            intent = "ingest"
        elif any(key in question for key in ("embed", "refresh vectors", "re-embed")):
            intent = "embed"
        elif any(key in question for key in ("status", "stats", "summary status")):
            intent = "status"
        elif any(key in question for key in ("show sources", "retrieve", "context only")):
            intent = "retrieve"
        else:
            intent = "answer"

        logger.info("Agent intent: %s", intent)
        state["intent"] = intent
        return state

    def _route_from_plan(self, state: AgentState) -> str:
        intent = state.get("intent", "answer")
        return intent

    def _ingest_node(self, state: AgentState) -> AgentState:
        ingest_data(self.raw_dir, self.processed_dir)
        embed_and_store(
            processed_file=str(Path(self.processed_dir) / "documents.json"),
            vector_store_path=self.vector_store_path,
        )
        state.setdefault("actions", []).extend(["ingest", "embed"])
        state["message"] = "📥 Ingestion and embedding completed. Knowledge base refreshed."
        doc_count = self._count_processed_documents()
        self._update_status(event="ingest", document_count=doc_count)
        return state

    def _embed_node(self, state: AgentState) -> AgentState:
        embed_and_store(
            processed_file=str(Path(self.processed_dir) / "documents.json"),
            vector_store_path=self.vector_store_path,
        )
        state.setdefault("actions", []).append("embed")
        state["message"] = "🧠 Embeddings updated from existing processed documents."
        doc_count = self._count_processed_documents()
        self._update_status(event="embed", document_count=doc_count)
        return state

    def _retrieve_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        results = query_vector_store(
            query=question,
            vector_store_path=self.vector_store_path,
            n_results=self.k,
        )
        results = self._rerank(question, results)
        state["context"] = results
        state.setdefault("actions", []).append("retrieve")
        state["message"] = self._format_context_preview(results)
        state["verification"] = self._basic_verification(question, results)
        return state

    def _answer_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        results = query_vector_store(
            query=question,
            vector_store_path=self.vector_store_path,
            n_results=self.k,
        )
        results = self._rerank(question, results)
        payload = answer_question(
            question=question,
            vector_store_path=self.vector_store_path,
            n_results=self.k,
            chat_history=self.history,
        )
        state["context"] = results
        state["answer"] = payload["answer"]
        state.setdefault("actions", []).extend(["retrieve", "answer"])
        state["message"] = payload["answer"]
        state["verification"] = ""
        return state

    def _status_node(self, state: AgentState) -> AgentState:
        status = self._read_status()
        state.setdefault("actions", []).append("status")
        if not status:
            state["message"] = (
                "No status information recorded yet. Run ingestion or embedding first."
            )
            return state

        lines = [
            "📊 Pipeline status:",
            f"  • Last ingestion: {status.get('last_ingest_at', 'never')}",
            f"  • Last embedding: {status.get('last_embed_at', 'never')}",
            f"  • Documents indexed: {status.get('documents_indexed', 'unknown')}",
            f"  • Vector store path: {status.get('vector_store_path', self.vector_store_path)}",
        ]
        state["message"] = "\n".join(lines)
        return state

    def _count_processed_documents(self) -> Optional[int]:
        processed_file = Path(self.processed_dir) / "documents.json"
        if not processed_file.exists():
            return None
        try:
            with processed_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return len(data)
        except Exception as exc:
            logger.warning("Failed to count processed documents: %s", exc)
            return None

    def _update_status(self, event: str, document_count: Optional[int] = None) -> None:
        status = {}
        if self.status_file.exists():
            try:
                with self.status_file.open("r", encoding="utf-8") as fh:
                    status = json.load(fh)
            except Exception:
                status = {}

        timestamp = datetime.utcnow().isoformat() + "Z"
        if event == "ingest":
            status["last_ingest_at"] = timestamp
        if event in {"ingest", "embed"}:
            status["last_embed_at"] = timestamp
        if document_count is not None:
            status["documents_indexed"] = document_count

        status["raw_dir"] = self.raw_dir
        status["processed_dir"] = self.processed_dir
        status["vector_store_path"] = self.vector_store_path

        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        with self.status_file.open("w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=2)

    def _read_status(self) -> Optional[Dict[str, object]]:
        if not self.status_file.exists():
            return None
        try:
            with self.status_file.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("Failed to read status file: %s", exc)
            return None

    def _log_interaction(self, state: AgentState) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "question": state.get("question"),
            "answer": state.get("answer") or state.get("message"),
            "actions": state.get("actions"),
            "verification": state.get("verification"),
            "sources": [
                {
                    "source": (item.get("metadata") or {}).get("source"),
                    "source_name": (item.get("metadata") or {}).get("source_name"),
                    "score": item.get("score"),
                }
                for item in state.get("context", []) or []
            ],
        }

        self.logs_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.logs_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record))
                fh.write("\n")
        except Exception as exc:
            logger.warning("Failed to append conversation log: %s", exc)

    def _verify_node(self, state: AgentState) -> AgentState:
        question = state.get("question", "")
        context = state.get("context", [])
        answer = state.get("answer")

        baseline = self._basic_verification(question, context)
        if not answer:
            state["verification"] = baseline
            return state

        self_check = self._self_review(answer, context)
        if baseline and self_check:
            state["verification"] = f"{baseline}\n{self_check}"
        else:
            state["verification"] = baseline or self_check
        return state

    def _finalize_node(self, state: AgentState) -> AgentState:
        if "message" not in state and "answer" in state:
            state["message"] = state["answer"]
        self._log_interaction(state)
        return state

    # Utility methods ------------------------------------------------------------
    @staticmethod
    def _format_context_preview(results: List[Dict[str, object]]) -> str:
        if not results:
            return "I could not find any matching context for that request."
        lines = ["Top retrieved passages:"]
        for idx, item in enumerate(results, start=1):
            metadata = item.get("metadata") or {}
            source = metadata.get("source_name") or metadata.get("source") or "unknown source"
            snippet = (item.get("content") or "").strip().replace("\n", " ")
            if len(snippet) > 160:
                snippet = f"{snippet[:157].rstrip()}…"
            score = item.get("score")
            score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
            lines.append(f"{idx}. {source} (score: {score_text}) – {snippet}")
        return "\n".join(lines)

    def _basic_verification(
        self,
        question: str,
        sources: List[Dict[str, object]],
    ) -> str:
        if not sources:
            return "⚠️ Verification: No supporting context retrieved."

        tokens = {
            token.lower()
            for token in question.split()
            if len(token) >= 4 and token.isalpha()
        }
        covered = set()
        for item in sources:
            content = (item.get("content") or "").lower()
            for token in tokens:
                if token in content:
                    covered.add(token)
        missing = sorted(tokens - covered)
        if missing:
            return (
                "⚠️ Verification: Some query terms were not found in the retrieved context "
                f"({', '.join(missing)})."
            )
        return "✅ Verification: Retrieved context covers the main query terms."

    def _self_review(self, answer: str, sources: List[Dict[str, object]]) -> str:
        context_blob = " ".join(
            (item.get("content") or "").lower() for item in sources if item
        ).strip()
        if not context_blob:
            return "⚠️ Self-check: No context available to verify the answer."

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", answer)
            if sentence.strip()
        ]
        issues: List[str] = []

        for sentence in sentences:
            if len(sentence) < 24:
                continue
            tokens = [
                token.lower()
                for token in re.findall(r"[A-Za-z]{4,}", sentence)
                if token.lower() not in self._STOPWORDS
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

    def _load_cross_encoder(self, model_name: Optional[str]):
        if not model_name:
            return None
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder model '%s' for reranking.", model_name)
            return CrossEncoder(model_name=model_name)
        except Exception as exc:
            logger.warning(
                "Failed to load cross-encoder '%s': %s. Continuing without reranker.",
                model_name,
                exc,
            )
            return None

    def _rerank(self, question: str, docs: List[Dict[str, object]]):
        if not self.cross_encoder or not docs:
            return docs
        try:
            pairs = [(question, (doc.get("content") or "")) for doc in docs]
            scores = self.cross_encoder.predict(pairs)
            ranked = [
                doc
                for _, doc in sorted(
                    zip(scores, docs), key=lambda item: item[0], reverse=True
                )
            ]
            return ranked
        except Exception as exc:
            logger.warning("Cross-encoder rerank failed: %s", exc)
            return docs
