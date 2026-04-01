"""
RETRIEVAL PIPELINE MODULE
==========================
Query understanding → multi-method search → post-retrieval processing.
"""

import re
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.embedding.engine import EmbeddingEngine, VectorStore
from src.utils.logger import log


@dataclass
class RetrievalResult:
    """A retrieved chunk with relevance scoring."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    source: str = ""
    chunk_id: str = ""


@dataclass
class QueryContext:
    """Enriched query with understanding signals."""
    original_query: str
    rewritten_query: str = ""
    entities: List[str] = field(default_factory=list)
    intent: str = ""
    expanded_terms: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)


class QueryUnderstanding:
    """Analyze and enrich user queries before retrieval."""

    def __init__(self, config: dict):
        self.config = config
        self.rewrite = config.get("rewrite_query", True)
        self.detect_entities = config.get("detect_entities", True)
        self.detect_intent = config.get("detect_intent", True)
        self.expand = config.get("expand_query", True)

    def process(self, query: str, chat_history: List[str] = None) -> QueryContext:
        """Build a QueryContext from a raw user query."""
        ctx = QueryContext(original_query=query)

        if self.rewrite and chat_history:
            ctx.rewritten_query = self._rewrite_with_context(query, chat_history)
        else:
            ctx.rewritten_query = query

        if self.detect_entities:
            ctx.entities = self._extract_entities(query)

        if self.detect_intent:
            ctx.intent = self._classify_intent(query)

        if self.expand:
            ctx.expanded_terms = self._expand_query(query)

        return ctx

    @staticmethod
    def _rewrite_with_context(query: str, history: List[str]) -> str:
        """Incorporate chat history context into the query."""
        recent = " ".join(history[-3:]) if history else ""
        if recent:
            return f"{query} (context: {recent[:200]})"
        return query

    @staticmethod
    def _extract_entities(query: str) -> List[str]:
        entities = []
        # Quoted terms
        entities.extend(re.findall(r'"([^"]+)"', query))
        # Capitalized words (potential proper nouns)
        entities.extend(re.findall(r'\b([A-Z][a-z]{2,}(?:\s[A-Z][a-z]+)*)\b', query))
        return list(set(entities))

    @staticmethod
    def _classify_intent(query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["how", "explain", "describe", "what is"]):
            return "informational"
        elif any(w in q for w in ["compare", "difference", "vs", "versus"]):
            return "comparison"
        elif any(w in q for w in ["list", "show", "give me", "find"]):
            return "listing"
        elif any(w in q for w in ["why", "reason", "cause"]):
            return "causal"
        elif any(w in q for w in ["should", "recommend", "best", "suggest"]):
            return "recommendation"
        return "general"

    @staticmethod
    def _expand_query(query: str) -> List[str]:
        """Generate synonyms / related terms."""
        expansions = {
            "ml": ["machine learning"],
            "ai": ["artificial intelligence"],
            "nlp": ["natural language processing"],
            "llm": ["large language model"],
            "rag": ["retrieval augmented generation"],
            "db": ["database"],
            "api": ["application programming interface"],
            "auth": ["authentication", "authorization"],
        }
        terms = []
        for word in query.lower().split():
            if word in expansions:
                terms.extend(expansions[word])
        return terms


class RetrievalEngine:
    """Orchestrates query → search → post-retrieval pipeline."""

    def __init__(self, config: dict, embedding_engine: EmbeddingEngine,
                 vector_store: VectorStore):
        self.config = config
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.query_understanding = QueryUnderstanding(
            config.get("query_understanding", {})
        )
        self.search_cfg = config.get("search", {})
        self.post_cfg = config.get("post_retrieval", {})

    def retrieve(self, query: str, chat_history: List[str] = None,
                 filters: Dict[str, Any] = None) -> List[RetrievalResult]:
        """Full retrieval pipeline: understand → search → post-process."""
        log.info(f"═══ RETRIEVAL: '{query[:80]}...' ═══")

        # 1. Query Understanding
        ctx = self.query_understanding.process(query, chat_history)
        log.info(f"Intent: {ctx.intent} | Entities: {ctx.entities}")

        # 2. Search
        search_query = ctx.rewritten_query or query
        results = self._multi_search(search_query, filters)
        log.info(f"Raw search results: {len(results)}")

        # 3. Post-Retrieval Processing
        results = self._post_process(results, query)
        log.info(f"After post-processing: {len(results)} results")

        return results

    def _multi_search(self, query: str, filters: Optional[dict]) -> List[RetrievalResult]:
        """Execute multiple search strategies and merge results."""
        methods = self.search_cfg.get("methods", ["vector"])
        all_results = []

        if "vector" in methods:
            vector_results = self._vector_search(query, filters)
            all_results.extend(vector_results)

        if "keyword" in methods:
            keyword_results = self._keyword_search(query, filters)
            all_results.extend(keyword_results)

        # Deduplicate by content hash
        seen = set()
        unique = []
        for r in all_results:
            key = hash(r.content[:200])
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def _vector_search(self, query: str, filters: Optional[dict]) -> List[RetrievalResult]:
        """Semantic vector search."""
        top_k = self.search_cfg.get("vector_top_k", 20)

        # Embed the query
        from src.ingestion.engine import Chunk
        query_chunk = Chunk(id="query", content=query, document_id="query")
        embedded = self.embedding_engine.embed_chunks([query_chunk])
        if not embedded:
            return []
        query_vector = embedded[0].vector

        # Search
        hits = self.vector_store.search(query_vector, top_k=top_k, filters=filters)
        return [
            RetrievalResult(
                content=h.get("content", ""),
                metadata=h.get("metadata", {}),
                score=1 - h.get("distance", 0),  # Convert distance to similarity
                chunk_id=h.get("id", ""),
            )
            for h in hits
        ]

    def _keyword_search(self, query: str, filters: Optional[dict]) -> List[RetrievalResult]:
        """BM25/keyword search fallback (simplified TF-IDF matching)."""
        # This is a simplified keyword search using the vector store's metadata
        # In production, use Elasticsearch or a dedicated BM25 engine
        log.info("Keyword search — simplified implementation")
        return []

    def _post_process(self, results: List[RetrievalResult],
                      original_query: str) -> List[RetrievalResult]:
        """Apply post-retrieval refinements."""
        # Reranking
        if self.post_cfg.get("reranking", {}).get("enabled"):
            results = self._rerank(results, original_query)

        # Deduplication
        if self.post_cfg.get("deduplication"):
            results = self._deduplicate(results)

        # Context compression
        if self.post_cfg.get("context_compression"):
            results = self._compress(results, original_query)

        # Chunk stitching
        if self.post_cfg.get("chunk_stitching"):
            results = self._stitch_chunks(results)

        # Final top-k
        final_k = self.post_cfg.get("reranking", {}).get("top_k", 10)
        return results[:final_k]

    def _rerank(self, results: List[RetrievalResult],
                query: str) -> List[RetrievalResult]:
        """Cross-encoder reranking for improved relevance."""
        model_name = self.post_cfg.get("reranking", {}).get(
            "model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(model_name)
            pairs = [(query, r.content) for r in results]
            scores = reranker.predict(pairs)
            for r, score in zip(results, scores):
                r.score = float(score)
            results.sort(key=lambda x: x.score, reverse=True)
            log.info("Reranking complete")
        except ImportError:
            log.warning("sentence-transformers not installed — skipping reranking")
            results.sort(key=lambda x: x.score, reverse=True)
        return results

    @staticmethod
    def _deduplicate(results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove near-duplicate results."""
        unique = []
        seen_content = set()
        for r in results:
            content_key = r.content[:100].lower().strip()
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique.append(r)
        return unique

    @staticmethod
    def _compress(results: List[RetrievalResult],
                  query: str) -> List[RetrievalResult]:
        """Compress context by keeping only query-relevant sentences."""
        query_words = set(query.lower().split())
        for r in results:
            sentences = re.split(r'(?<=[.!?])\s+', r.content)
            relevant = [s for s in sentences
                        if any(w in s.lower() for w in query_words)]
            if relevant:
                r.content = " ".join(relevant)
        return results

    @staticmethod
    def _stitch_chunks(results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Merge adjacent chunks from the same document."""
        if len(results) < 2:
            return results
        
        merged = [results[0]]
        for r in results[1:]:
            prev = merged[-1]
            same_doc = (prev.metadata.get("document_id") == r.metadata.get("document_id"))
            adjacent = abs(
                prev.metadata.get("chunk_index", -1) - r.metadata.get("chunk_index", -2)
            ) == 1
            if same_doc and adjacent:
                prev.content += "\n" + r.content
                prev.score = max(prev.score, r.score)
            else:
                merged.append(r)
        return merged
