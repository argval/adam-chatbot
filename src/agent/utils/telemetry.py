import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


@dataclass
class TelemetryConfig:
    status_path: Path = field(
        default_factory=lambda: Path(os.getenv("AGENT_STATUS_PATH", "data/status.json"))
    )
    logs_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("AGENT_LOG_PATH", "data/logs/conversations.jsonl")
        )
    )
    log_rotation_bytes: int = int(os.getenv("AGENT_LOG_ROTATION_BYTES", "2000000"))


class TelemetryManager:
    def __init__(self, config: Optional[TelemetryConfig] = None) -> None:
        self.config = config or TelemetryConfig()

    # Status -----------------------------------------------------------------
    def update_status(
        self,
        *,
        event: str,
        raw_dir: str,
        processed_dir: str,
        vector_store_path: str,
        document_count: Optional[int] = None,
    ) -> None:
        try:
            status = self.read_status() or {}
            timestamp = datetime.utcnow().isoformat() + "Z"
            if event == "ingest":
                status["last_ingest_at"] = timestamp
            if event in {"ingest", "embed"}:
                status["last_embed_at"] = timestamp
            if document_count is not None:
                status["documents_indexed"] = document_count

            status["raw_dir"] = raw_dir
            status["processed_dir"] = processed_dir
            status["vector_store_path"] = vector_store_path

            self.config.status_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config.status_path.open("w", encoding="utf-8") as fh:
                json.dump(status, fh, indent=2)
        except Exception as exc:
            # Do not propagate telemetry issues to the agent loop.
            import logging

            logging.getLogger(__name__).warning("Failed to update status: %s", exc)

    def read_status(self) -> Optional[Dict[str, object]]:
        if not self.config.status_path.exists():
            return None
        try:
            with self.config.status_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    # Logging -----------------------------------------------------------------
    def log_interaction(self, record: Dict[str, object]) -> None:
        try:
            record = dict(record)
            record.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
            self._rotate_logs_if_needed()
            self.config.logs_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config.logs_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record))
                fh.write("\n")
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("Failed to append conversation log: %s", exc)

    def _rotate_logs_if_needed(self) -> None:
        log_path = self.config.logs_path
        if not log_path.exists():
            return
        if log_path.stat().st_size < self.config.log_rotation_bytes:
            return

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        backup_path = log_path.with_suffix(f".{timestamp}.bak")
        shutil.move(log_path, backup_path)

    # Utilities ---------------------------------------------------------------
    @staticmethod
    def count_processed_documents(processed_dir: str) -> Optional[int]:
        processed_file = Path(processed_dir) / "documents.json"
        if not processed_file.exists():
            return None
        try:
            with processed_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return len(data)
        except Exception:
            return None
