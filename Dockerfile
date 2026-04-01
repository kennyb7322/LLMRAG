# ============================================================================
# LLMRAG Dockerfile
# Multi-stage build for optimized container
# ============================================================================

FROM python:3.11-slim AS base

LABEL maintainer="Ken Barnes / UCS Solutions"
LABEL description="LLMRAG - Automated LLM & RAG Pipeline"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Create directories
RUN mkdir -p data/documents data/vectordb logs models/embeddings models/llm configs

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: start API server
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
