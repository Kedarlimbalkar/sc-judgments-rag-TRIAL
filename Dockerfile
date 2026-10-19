# ==============================================================================
# Supreme Court Judgments RAG - Cloud Run Container Definition
# Base: Python 3.11 Slim
# Target Port: 8080
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    APP_HOME=/app

# Install minimal OS dependencies for network health checks and database runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR ${APP_HOME}

# Create a non-privileged system user for container security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Install Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and schema
COPY config.py .
COPY schema.sql .
COPY online_app.py .
COPY offline_pipeline.py .

# Assign ownership to the non-privileged user
RUN chown -R appuser:appgroup ${APP_HOME}

# Switch to non-root execution
USER appuser

# Expose Cloud Run default port
EXPOSE 8080

# Health check probe for orchestrators
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/healthz || exit 1

# Execute FastAPI via Uvicorn with dynamic Cloud Run PORT binding
CMD ["sh", "-c", "exec uvicorn online_app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 2"]

