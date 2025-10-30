from __future__ import annotations

import logging
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from src.agent.orchestrator import AgentOrchestrator, AgentConfig
from src.agent.status import get_status
from src.agent.utils import TelemetryConfig
from src.embeddings.embed import embed_and_store
from src.ingestion.ingest import ingest_data

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    message: str
    verification: Optional[str]
    sources: list


class IngestRequest(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    chunk_size: int = 2200
    chunk_overlap: int = 220


class EmbedRequest(BaseModel):
    processed_file: str = "data/processed/documents.json"
    vector_store_path: str = "data/vector_store"
    reset: bool = False


class StatusResponse(BaseModel):
    status: dict


def create_agent() -> AgentOrchestrator:
    config = AgentConfig()
    return AgentOrchestrator(config=config)


def create_app() -> FastAPI:
    app = FastAPI(title="Adam Chatbot API", version="1.0.0")

    agent_dependency = Depends(create_agent)

    @app.post("/ask", response_model=AskResponse)
    def ask_endpoint(
        payload: AskRequest,
        agent: Annotated[AgentOrchestrator, agent_dependency],
    ) -> AskResponse:
        result = agent.handle(payload.question)
        if "message" not in result:
            raise HTTPException(status_code=500, detail="Agent returned no message")
        return AskResponse(
            message=result["message"],
            verification=result.get("verification"),
            sources=result.get("context", []),
        )

    @app.post("/ingest")
    def ingest_endpoint(payload: IngestRequest):
        stats = ingest_data(
            raw_dir=payload.raw_dir,
            processed_dir=payload.processed_dir,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
        )
        return {
            "processed": stats.processed_files,
            "skipped": stats.skipped_files,
            "errors": stats.error_files,
            "chunks": stats.generated_chunks,
        }

    @app.post("/embed")
    def embed_endpoint(payload: EmbedRequest):
        embed_and_store(
            processed_file=payload.processed_file,
            vector_store_path=payload.vector_store_path,
            reset=payload.reset,
        )
        return {"status": "ok"}

    @app.get("/status", response_model=StatusResponse)
    def status_endpoint():
        status = get_status(TelemetryConfig())
        if status is None:
            raise HTTPException(status_code=404, detail="Status not available")
        return StatusResponse(status=status)

    @app.get("/healthz")
    def health_endpoint():
        return {"status": "ok"}

    return app


app = create_app()
