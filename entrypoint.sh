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

# Otherwise, default to running the Streamlit web application
echo "[Entrypoint] Launching Supreme Court RAG Streamlit app on port ${PORT:-8080}..."
exec streamlit run app.py --server.port="${PORT:-8080}" --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false

