#!/bin/sh
set -e

# Automatically detect if running inside a Google Cloud Run Job
# Cloud Run automatically injects CLOUD_RUN_JOB environment variable into job tasks.
if [ -n "$CLOUD_RUN_JOB" ] || [ "$MODE" = "offline" ] || [ "$MODE" = "batch" ]; then
    echo "[Entrypoint] Detected Cloud Run Job environment ($CLOUD_RUN_JOB) - Launching Offline Ingestion Pipeline..."
    exec python offline_pipeline.py "$@"
fi

# If explicit command arguments were passed, execute them directly
if [ $# -gt 0 ]; then
    exec "$@"
fi

# Otherwise, default to running the FastAPI online serving application
echo "[Entrypoint] Launching Supreme Court RAG FastAPI server on port ${PORT:-8080}..."
exec uvicorn online_app:app --host 0.0.0.0 --port "${PORT:-8080}" --workers 2

