import logging
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.embeddings.embed import embed_and_store
from src.ingestion.ingest import ingest_data
from src.llm.qa import answer_question
from src.vector_store.query import query_vector_store
from .utils import (
    TelemetryConfig,
    TelemetryManager,
    VerificationConfig,
    VerificationHelper,
)

load_dotenv()

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class AgentConfig:
    top_k: int = int(os.getenv("AGENT_TOP_K", "5"))
    rerank_enabled: bool = _env_flag("AGENT_RERANK_ENABLED", True)
    cross_encoder_model: Optional[str] = os.getenv("AGENT_RERANK_MODEL")


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
    raw_dir: str = field(default_factory=lambda: os.getenv("AGENT_RAW_DIR", "data/raw"))
    processed_dir: str = field(
        default_factory=lambda: os.getenv("AGENT_PROCESSED_DIR", "data/processed")
    )
    vector_store_path: str = field(
        default_factory=lambda: os.getenv("AGENT_VECTOR_STORE_PATH", "data/vector_store")
    )
    config: AgentConfig = field(default_factory=AgentConfig)
    telemetry_config: TelemetryConfig = field(default_factory=TelemetryConfig)
    verification_config: VerificationConfig = field(default_factory=VerificationConfig)
    history: List[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.top_k = self.config.top_k
        self.rerank_enabled = self.config.rerank_enabled
        self.cross_encoder = (
            self._load_cross_encoder(self.config.cross_encoder_model)
            if self.rerank_enabled and self.config.cross_encoder_model
            else None
        )
        self.telemetry = TelemetryManager(self.telemetry_config)
        self.verifier = VerificationHelper(self.verification_config)
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
        doc_count = TelemetryManager.count_processed_documents(self.processed_dir)
        self.telemetry.update_status(
            event="ingest",
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
            vector_store_path=self.vector_store_path,
            document_count=doc_count,
        )
        return state

    def _embed_node(self, state: AgentState) -> AgentState:
        embed_and_store(
            processed_file=str(Path(self.processed_dir) / "documents.json"),
            vector_store_path=self.vector_store_path,
        )
        state.setdefault("actions", []).append("embed")
        state["message"] = "🧠 Embeddings updated from existing processed documents."
        doc_count = TelemetryManager.count_processed_documents(self.processed_dir)
        self.telemetry.update_status(
            event="embed",
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
            vector_store_path=self.vector_store_path,
            document_count=doc_count,
        )
        return state

    def _retrieve_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        results = query_vector_store(
            query=question,
            vector_store_path=self.vector_store_path,
            n_results=self.top_k,
        )
        results = self._rerank(question, results)
        state["context"] = results
        state.setdefault("actions", []).append("retrieve")
        state["message"] = self._format_context_preview(results)
        state["verification"] = self.verifier.basic(question, results)
        return state

    def _answer_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        results = query_vector_store(
            query=question,
            vector_store_path=self.vector_store_path,
            n_results=self.top_k,
        )
        results = self._rerank(question, results)
        payload = answer_question(
            question=question,
            vector_store_path=self.vector_store_path,
            n_results=self.top_k,
            chat_history=self.history,
        )
        state["context"] = results
        state["answer"] = payload["answer"]
        state.setdefault("actions", []).extend(["retrieve", "answer"])
        state["message"] = payload["answer"]
        state["verification"] = ""
        return state

    def _status_node(self, state: AgentState) -> AgentState:
        status = self.telemetry.read_status()
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

    def _verify_node(self, state: AgentState) -> AgentState:
        question = state.get("question", "")
        context = state.get("context", [])
        answer = state.get("answer")

        baseline = self.verifier.basic(question, context)
        if not answer:
            state["verification"] = baseline
            return state

        self_check = self.verifier.self_check(answer, context)
        if baseline and self_check:
            state["verification"] = f"{baseline}\n{self_check}"
        else:
            state["verification"] = baseline or self_check
        return state

    def _finalize_node(self, state: AgentState) -> AgentState:
        if "message" not in state and "answer" in state:
            state["message"] = state["answer"]
        sources = [
            {
                "source": (item.get("metadata") or {}).get("source"),
                "source_name": (item.get("metadata") or {}).get("source_name"),
                "score": item.get("score"),
            }
            for item in state.get("context", []) or []
        ]
        self.telemetry.log_interaction(
            {
                "question": state.get("question"),
                "answer": state.get("answer") or state.get("message"),
                "actions": state.get("actions"),
                "verification": state.get("verification"),
                "sources": sources,
            }
        )
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
        if not self.rerank_enabled or not self.cross_encoder or not docs:
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
