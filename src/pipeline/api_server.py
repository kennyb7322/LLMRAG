"""
LLMRAG API Server
===================
FastAPI-based REST API for the LLM+RAG pipeline.
"""

import os
import time
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from src.pipeline.orchestrator import LLMRAGPipeline
from src.utils.logger import log

# Pipeline instance (initialized on startup)
pipeline: Optional[LLMRAGPipeline] = None


def create_app():
    """Create and configure the FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException, Header, Depends
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        log.error("FastAPI not installed — pip install 'fastapi[standard]'")
        raise

    # ── Models ──────────────────────────────────────────────────────────────

    class QueryRequest(BaseModel):
        question: str
        chat_history: List[str] = []
        filters: Dict[str, Any] = {}

    class QueryResponse(BaseModel):
        answer: str
        citations: List[Dict[str, Any]] = []
        sources: List[str] = []
        confidence: float = 0.0
        tokens_used: int = 0

    class IngestRequest(BaseModel):
        config_override: Dict[str, Any] = {}

    class HealthResponse(BaseModel):
        status: str
        components: Dict[str, str] = {}
        stats: Dict[str, Any] = {}

    # ── App ─────────────────────────────────────────────────────────────────

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global pipeline
        config_path = os.environ.get("LLMRAG_CONFIG", "configs/pipeline_config.yaml")
        pipeline = LLMRAGPipeline(config_path)
        log.info("Pipeline initialized — API ready")
        yield
        log.info("Shutting down")

    app = FastAPI(
        title="LLMRAG API",
        description="Automated LLM & RAG Deployment Framework",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Auth Dependency ─────────────────────────────────────────────────────

    async def verify_auth(x_api_key: str = Header(default="")):
        if pipeline and pipeline.security.config.get("authentication", {}).get("enabled"):
            user = pipeline.security.authenticate_request({"X-API-Key": x_api_key})
            if not user:
                raise HTTPException(status_code=401, detail="Invalid API key")
            return user
        return {"role": "admin", "name": "auth_disabled"}

    # ── Endpoints ───────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return pipeline.health_check()

    @app.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest, user=Depends(verify_auth)):
        if not pipeline:
            raise HTTPException(500, "Pipeline not initialized")
        
        result = pipeline.query(
            question=req.question,
            chat_history=req.chat_history,
            filters=req.filters,
            user=user.get("name", "anonymous"),
        )
        return QueryResponse(
            answer=result.answer,
            citations=result.citations,
            sources=result.sources,
            confidence=result.confidence,
            tokens_used=result.tokens_used,
        )

    @app.post("/ingest")
    async def ingest(user=Depends(verify_auth)):
        if not pipeline:
            raise HTTPException(500, "Pipeline not initialized")
        if user.get("role") not in ("admin",):
            raise HTTPException(403, "Admin role required for ingestion")
        
        stats = pipeline.ingest()
        return {
            "status": "complete",
            "chunks": stats.chunks_created,
            "vectors": stats.vectors_stored,
            "time_sec": round(stats.ingestion_time_sec + stats.embedding_time_sec, 2),
        }

    @app.get("/compliance")
    async def compliance(user=Depends(verify_auth)):
        return pipeline.compliance_report()

    @app.get("/stats")
    async def stats():
        s = pipeline.stats
        return {
            "documents_ingested": s.documents_ingested,
            "chunks_created": s.chunks_created,
            "vectors_stored": s.vectors_stored,
            "queries_processed": s.queries_processed,
            "total_tokens_used": s.total_tokens_used,
            "avg_query_time_sec": round(s.avg_query_time_sec, 3),
        }

    return app


# Run directly: python -m src.pipeline.api_server
if __name__ == "__main__":
    import uvicorn
    app = create_app()
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
