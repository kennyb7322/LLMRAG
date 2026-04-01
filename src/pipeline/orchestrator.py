"""
PIPELINE ORCHESTRATOR — V3
============================
Full-stack engine wiring:
  Ingestion → Embedding → Retrieval → Generation → Security
  + Cloud Providers (Azure/AWS/GCP/OCI)
  + AGI Cognitive Architecture (8 layers)
  + Agentic Framework (6 agent types)
  + Streaming (SSE token-by-token)
  + Multi-Modal (images, tables, PDFs)
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.utils.config import PipelineConfig, load_config
from src.utils.logger import log, setup_logger
from src.ingestion.engine import IngestionEngine, Chunk
from src.embedding.engine import EmbeddingEngine, VectorStore
from src.retrieval.engine import RetrievalEngine
from src.generation.engine import GenerationEngine, GenerationOutput
from src.security.engine import SecurityEngine
from src.cloud.providers import CloudProviderFactory
from src.agi.framework import AGIOrchestrator
from src.agentic.agents import AgentOrchestrator, RAGAgent, ResearchAgent, ToolRegistry
from src.streaming.engine import StreamingEngine
from src.multimodal.engine import MultiModalEngine


@dataclass
class PipelineStats:
    """Runtime statistics for the pipeline."""
    documents_ingested: int = 0
    chunks_created: int = 0
    vectors_stored: int = 0
    queries_processed: int = 0
    total_tokens_used: int = 0
    ingestion_time_sec: float = 0.0
    embedding_time_sec: float = 0.0
    avg_query_time_sec: float = 0.0
    cloud_providers_active: int = 0
    agi_enabled: bool = False
    agents_registered: int = 0


class LLMRAGPipeline:
    """
    Main Pipeline Orchestrator
    ===========================
    Usage:
        pipeline = LLMRAGPipeline("configs/pipeline_config.yaml")
        pipeline.ingest()          # Load, chunk, embed, store
        result = pipeline.query("What is ...?")  # RAG query
        print(result.answer)
    """

    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        
        # Setup logging
        obs = self.config.observability
        setup_logger(
            level=obs.get("logging", {}).get("level", "INFO"),
            log_file=obs.get("logging", {}).get("file"),
            json_format=obs.get("logging", {}).get("format") == "json",
        )

        # ── Core RAG Engines ───────────────────────────────────────────
        self.ingestion = IngestionEngine(self.config.ingestion)
        self.embedding_engine = EmbeddingEngine(self.config.embedding)
        self.vector_store = VectorStore(self.config.embedding)
        self.retrieval = RetrievalEngine(
            self.config.retrieval,
            self.embedding_engine,
            self.vector_store,
        )
        self.generation = GenerationEngine(self.config.generation)
        self.security = SecurityEngine(self.config.security)

        # ── Cloud Providers (Azure/AWS/GCP/OCI) ────────────────────────
        cloud_cfg = self.config.raw.get("cloud", {})
        self.cloud = CloudProviderFactory(cloud_cfg)
        self.cloud.initialize_all()

        # ── AGI Cognitive Architecture ─────────────────────────────────
        agi_cfg = self.config.raw.get("agi", {})
        self.agi = AGIOrchestrator(agi_cfg)

        # ── Agentic Framework ──────────────────────────────────────────
        agentic_cfg = self.config.raw.get("agentic", {})
        self.agents = AgentOrchestrator(agentic_cfg)
        self._register_default_agents(agentic_cfg)

        # ── Streaming Engine ──────────────────────────────────────────
        self.streaming = StreamingEngine(self.config.generation)

        # ── Multi-Modal Engine ─────────────────────────────────────────
        mm_cfg = self.config.raw.get("multimodal", {})
        self.multimodal = MultiModalEngine(mm_cfg)

        # Stats
        self.stats = PipelineStats(
            cloud_providers_active=len(self.cloud.providers),
            agi_enabled=agi_cfg.get("enabled", False),
            agents_registered=len(self.agents.agents),
        )

        log.info("╔══════════════════════════════════════════════════════╗")
        log.info("║   LLMRAG-V3 Pipeline Initialized                   ║")
        log.info("║   LLM + RAG + Cloud + AGI + Agentic + Streaming    ║")
        log.info(f"║   Cloud: {len(self.cloud.providers)} | "
                 f"Agents: {len(self.agents.agents)} | "
                 f"AGI: {'ON' if agi_cfg.get('enabled') else 'OFF':>3}       ║")
        log.info("╚══════════════════════════════════════════════════════╝")

    def _register_default_agents(self, config: dict):
        """Register built-in agents."""
        rag_agent = RAGAgent(config, pipeline=self)
        research_agent = ResearchAgent(config)
        self.agents.register_agent(rag_agent)
        self.agents.register_agent(research_agent)

    # ── Ingestion ───────────────────────────────────────────────────────────

    def ingest(self) -> PipelineStats:
        """
        Run the full ingestion pipeline:
        Load data → preprocess → chunk → enrich → embed → store.
        """
        log.info("\n🔄 STARTING FULL INGESTION PIPELINE")
        total_start = time.time()

        # Step 1: Ingest and chunk
        t0 = time.time()
        chunks = self.ingestion.run()
        self.stats.ingestion_time_sec = time.time() - t0
        self.stats.chunks_created = len(chunks)

        if not chunks:
            log.warning("No chunks generated — check data sources in config.")
            return self.stats

        # Step 2: Embed
        t0 = time.time()
        embedded = self.embedding_engine.embed_chunks(chunks)
        self.stats.embedding_time_sec = time.time() - t0
        self.stats.vectors_stored = len(embedded)

        # Step 3: Store
        self.vector_store.initialize()
        self.vector_store.store(embedded)

        # Step 4: Audit log
        self.security.audit.log_ingestion(
            source="pipeline",
            doc_count=self.stats.documents_ingested,
            chunk_count=self.stats.chunks_created,
        )

        total_time = time.time() - total_start
        log.info(f"\n✅ INGESTION COMPLETE in {total_time:.1f}s")
        log.info(f"   Chunks: {self.stats.chunks_created}")
        log.info(f"   Vectors: {self.stats.vectors_stored}")
        
        return self.stats

    # ── Query ───────────────────────────────────────────────────────────────

    def query(self, question: str,
              chat_history: List[str] = None,
              filters: Dict[str, Any] = None,
              user: str = "anonymous") -> GenerationOutput:
        """
        Run a RAG query:
        Query understanding → retrieval → generation → output.
        """
        t0 = time.time()

        # Security: log query
        self.security.log_query(question, user)

        # Step 1: Retrieve
        results = self.retrieval.retrieve(question, chat_history, filters)

        # Step 2: Generate
        output = self.generation.generate(question, results)

        # Step 3: Log response
        self.security.log_response(question, output.answer, user)

        # Stats
        self.stats.queries_processed += 1
        self.stats.total_tokens_used += output.tokens_used
        query_time = time.time() - t0
        self.stats.avg_query_time_sec = (
            (self.stats.avg_query_time_sec * (self.stats.queries_processed - 1)
             + query_time) / self.stats.queries_processed
        )

        log.info(f"Query completed in {query_time:.2f}s | "
                 f"Confidence: {output.confidence:.0%}")

        return output

    # ── Batch Query ─────────────────────────────────────────────────────────

    def batch_query(self, questions: List[str]) -> List[GenerationOutput]:
        """Process multiple queries."""
        return [self.query(q) for q in questions]

    # ── Health Check ────────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        """Check pipeline component health."""
        return {
            "status": "healthy",
            "version": "3.0.0",
            "components": {
                "ingestion": "ready",
                "embedding": f"{self.embedding_engine.provider}",
                "vector_store": f"{self.vector_store.provider}",
                "generation": f"{self.generation.provider}/{self.generation.model}",
                "security": "enabled" if self.security.config.get(
                    "authentication", {}
                ).get("enabled") else "disabled",
                "streaming": "ready",
                "multimodal": "ready",
            },
            "cloud_providers": self.cloud.health_check_all(),
            "agi": {
                "enabled": self.stats.agi_enabled,
                "layers": 8,
                "reasoning": self.agi.cfg.reasoning_strategy if self.stats.agi_enabled else "disabled",
            },
            "agentic": self.agents.health_check(),
            "stats": {
                "chunks_stored": self.stats.vectors_stored,
                "queries_processed": self.stats.queries_processed,
                "total_tokens": self.stats.total_tokens_used,
                "cloud_active": self.stats.cloud_providers_active,
                "agents": self.stats.agents_registered,
            },
        }

    # ── Compliance Report ───────────────────────────────────────────────────

    def compliance_report(self) -> Dict[str, Any]:
        """Run compliance checks."""
        return self.security.check_compliance(self.config.raw)
