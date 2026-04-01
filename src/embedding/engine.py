"""
EMBEDDING + STORAGE MODULE
===========================
Vectorize chunks and store in vector/graph databases with caching.
"""

import os
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.ingestion.engine import Chunk
from src.utils.logger import log


@dataclass
class EmbeddingResult:
    """An embedded chunk ready for storage."""
    chunk_id: str
    vector: List[float]
    content: str
    metadata: Dict[str, Any]


class EmbeddingEngine:
    """Generates embeddings from text chunks using configurable providers."""

    def __init__(self, config: dict):
        self.config = config
        self.model_cfg = config.get("model", {})
        self.provider = self.model_cfg.get("provider", "openai")
        self.model_name = self.model_cfg.get("name", "text-embedding-3-small")
        self.dimensions = self.model_cfg.get("dimensions", 1536)
        self.batch_size = self.model_cfg.get("batch_size", 100)
        self._client = None
        self._cache = {}

    def embed_chunks(self, chunks: List[Chunk]) -> List[EmbeddingResult]:
        """Embed a list of chunks, returning EmbeddingResults."""
        log.info(f"═══ EMBEDDING {len(chunks)} CHUNKS ═══")
        log.info(f"Provider: {self.provider} | Model: {self.model_name}")

        results = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            texts = [c.content for c in batch]
            vectors = self._embed_batch(texts)
            
            for chunk, vector in zip(batch, vectors):
                results.append(EmbeddingResult(
                    chunk_id=chunk.id,
                    vector=vector,
                    content=chunk.content,
                    metadata=chunk.metadata,
                ))
            
            log.info(f"Embedded batch {i // self.batch_size + 1} "
                     f"({len(results)}/{len(chunks)})")

        log.info(f"═══ EMBEDDING COMPLETE: {len(results)} vectors ═══")
        return results

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Route to the correct embedding provider."""
        # Check cache first
        uncached_indices = []
        results = [None] * len(texts)
        
        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached_indices.append(i)

        if not uncached_indices:
            return results

        uncached_texts = [texts[i] for i in uncached_indices]

        if self.provider == "openai":
            vectors = self._embed_openai(uncached_texts)
        elif self.provider == "huggingface":
            vectors = self._embed_huggingface(uncached_texts)
        elif self.provider == "cohere":
            vectors = self._embed_cohere(uncached_texts)
        elif self.provider == "local":
            vectors = self._embed_local(uncached_texts)
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

        for idx, vec in zip(uncached_indices, vectors):
            results[idx] = vec
            cache_key = hashlib.md5(texts[idx].encode()).hexdigest()
            self._cache[cache_key] = vec

        return results

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            response = client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except ImportError:
            log.error("openai not installed — pip install openai")
            return [[0.0] * self.dimensions] * len(texts)

    def _embed_huggingface(self, texts: List[str]) -> List[List[float]]:
        try:
            from sentence_transformers import SentenceTransformer
            if not self._client:
                self._client = SentenceTransformer(self.model_name)
            embeddings = self._client.encode(texts, show_progress_bar=False)
            return [e.tolist() for e in embeddings]
        except ImportError:
            log.error("sentence-transformers not installed")
            return [[0.0] * self.dimensions] * len(texts)

    def _embed_cohere(self, texts: List[str]) -> List[List[float]]:
        try:
            import cohere
            client = cohere.Client(os.environ.get("COHERE_API_KEY", ""))
            response = client.embed(texts=texts, model=self.model_name, input_type="search_document")
            return response.embeddings
        except ImportError:
            log.error("cohere not installed — pip install cohere")
            return [[0.0] * self.dimensions] * len(texts)

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        local_cfg = self.config.get("local_model", {})
        model_path = local_cfg.get("path", "./models/embeddings")
        try:
            from sentence_transformers import SentenceTransformer
            if not self._client:
                self._client = SentenceTransformer(model_path)
            embeddings = self._client.encode(texts, show_progress_bar=False)
            return [e.tolist() for e in embeddings]
        except ImportError:
            log.error("sentence-transformers not installed")
            return [[0.0] * self.dimensions] * len(texts)


# ── Vector Store Abstraction ────────────────────────────────────────────────

class VectorStore:
    """Unified interface to multiple vector database backends."""

    def __init__(self, config: dict):
        storage_cfg = config.get("storage", {})
        vdb_cfg = storage_cfg.get("vector_db", {})
        self.provider = vdb_cfg.get("provider", "chromadb")
        self.collection_name = vdb_cfg.get("collection_name", "llmragv3_vectors")
        self.vdb_config = vdb_cfg
        self._store = None

    def initialize(self):
        """Set up the vector store connection."""
        if self.provider == "chromadb":
            self._init_chromadb()
        elif self.provider == "pinecone":
            self._init_pinecone()
        elif self.provider == "pgvector":
            self._init_pgvector()
        elif self.provider == "qdrant":
            self._init_qdrant()
        else:
            log.warning(f"Vector store '{self.provider}' — using ChromaDB fallback")
            self._init_chromadb()

    def store(self, results: List[EmbeddingResult]):
        """Store embedding results in the vector database."""
        log.info(f"Storing {len(results)} vectors in {self.provider}")
        
        if self.provider == "chromadb":
            self._store_chromadb(results)
        elif self.provider == "pinecone":
            self._store_pinecone(results)
        elif self.provider == "pgvector":
            self._store_pgvector(results)
        elif self.provider == "qdrant":
            self._store_qdrant(results)
        
        log.info(f"Stored {len(results)} vectors successfully")

    def search(self, query_vector: List[float], top_k: int = 10,
               filters: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if self.provider == "chromadb":
            return self._search_chromadb(query_vector, top_k, filters)
        elif self.provider == "pinecone":
            return self._search_pinecone(query_vector, top_k, filters)
        return []

    # ── ChromaDB ────────────────────────────────────────────────────────────

    def _init_chromadb(self):
        try:
            import chromadb
            chroma_cfg = self.vdb_config.get("chromadb", {})
            persist_dir = chroma_cfg.get("persist_directory", "./data/vectordb")
            self._store = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._store.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": chroma_cfg.get("distance_metric", "cosine")},
            )
            log.info(f"ChromaDB initialized at {persist_dir}")
        except ImportError:
            log.error("chromadb not installed — pip install chromadb")

    def _store_chromadb(self, results: List[EmbeddingResult]):
        batch_size = 500
        for i in range(0, len(results), batch_size):
            batch = results[i:i + batch_size]
            self._collection.upsert(
                ids=[r.chunk_id for r in batch],
                embeddings=[r.vector for r in batch],
                documents=[r.content for r in batch],
                metadatas=[r.metadata for r in batch],
            )

    def _search_chromadb(self, query_vector, top_k, filters):
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filters,
        )
        hits = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                hits.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    "id": results["ids"][0][i] if results["ids"] else "",
                })
        return hits

    # ── Pinecone ────────────────────────────────────────────────────────────

    def _init_pinecone(self):
        try:
            from pinecone import Pinecone
            pc_cfg = self.vdb_config.get("pinecone", {})
            pc = Pinecone(api_key=pc_cfg.get("api_key") or os.environ.get("PINECONE_API_KEY"))
            index_name = pc_cfg.get("index_name", self.collection_name)
            self._store = pc.Index(index_name)
            log.info(f"Pinecone initialized: {index_name}")
        except ImportError:
            log.error("pinecone not installed — pip install pinecone")

    def _store_pinecone(self, results: List[EmbeddingResult]):
        vectors = [(r.chunk_id, r.vector, {**r.metadata, "content": r.content[:1000]})
                    for r in results]
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self._store.upsert(vectors=vectors[i:i + batch_size])

    def _search_pinecone(self, query_vector, top_k, filters):
        results = self._store.query(vector=query_vector, top_k=top_k,
                                     include_metadata=True, filter=filters)
        return [{"content": m.metadata.get("content", ""),
                 "metadata": m.metadata, "score": m.score, "id": m.id}
                for m in results.matches]

    # ── pgvector (stub) ────────────────────────────────────────────────────

    def _init_pgvector(self):
        log.info("pgvector store initialized (configure connection_string)")

    def _store_pgvector(self, results):
        log.warning("pgvector store not fully implemented — use ChromaDB or Pinecone")

    # ── Qdrant (stub) ──────────────────────────────────────────────────────

    def _init_qdrant(self):
        log.info("Qdrant store initialized")

    def _store_qdrant(self, results):
        log.warning("Qdrant store not fully implemented — use ChromaDB or Pinecone")
