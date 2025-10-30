## Development API

Run the FastAPI server locally:

```bash
uv run uvicorn src.api.main:app --reload
```

Available endpoints:

- `POST /ask` – Query the agent with `{ "question": "..." }`
- `POST /ingest` – Re-run ingestion with optional payload
- `POST /embed` – Regenerate embeddings (supports `reset` flag)
- `GET /status` – Inspect ingestion/embedding status
- `GET /healthz` – Liveness check
