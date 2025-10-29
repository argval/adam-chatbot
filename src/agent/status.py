"""Convenience helpers for reading agent telemetry status."""

from typing import Optional, Dict

from .utils import TelemetryManager, TelemetryConfig


def get_status(config: Optional[TelemetryConfig] = None) -> Optional[Dict[str, object]]:
    """Return the latest ingestion/embedding status if available."""
    manager = TelemetryManager(config)
    return manager.read_status()


if __name__ == "__main__":
    status = get_status()
    if status is None:
        print("No status file found.")
    else:
        import json

        print(json.dumps(status, indent=2))
