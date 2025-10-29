from pathlib import Path

from src.agent.status import get_status
from src.agent.utils import TelemetryConfig, TelemetryManager


def test_get_status_returns_keys(tmp_path: Path):
    config = TelemetryConfig(status_path=tmp_path / "status.json", logs_path=tmp_path / "logs.jsonl")
    manager = TelemetryManager(config)
    manager.update_status(
        event="ingest",
        raw_dir="data/raw",
        processed_dir="data/processed",
        vector_store_path="data/vector_store",
        document_count=10,
    )

    status = get_status(config)
    assert status is not None
    assert status["raw_dir"] == "data/raw"
    assert status["documents_indexed"] == 10
