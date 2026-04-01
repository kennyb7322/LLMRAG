"""
LLMRAG Test Suite
===================
Unit tests for all pipeline components.
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, PipelineConfig
from src.ingestion.engine import (
    Document, Chunk, Preprocessor, Chunker, Enricher,
    FileSystemLoader, IngestionEngine,
)
from src.security.engine import APIKeyManager, SecurityEngine


# ── Config Tests ────────────────────────────────────────────────────────────

class TestConfig:
    def test_load_defaults(self):
        config = PipelineConfig(raw={})
        assert config.ingestion == {}
        assert config.embedding == {}

    def test_dot_access(self):
        config = PipelineConfig(raw={"ingestion": {"sources": [{"type": "filesystem"}]}})
        assert config.get("ingestion.sources") == [{"type": "filesystem"}]
        assert config.get("missing.path", "default") == "default"


# ── Ingestion Tests ─────────────────────────────────────────────────────────

class TestPreprocessor:
    def setup_method(self):
        self.pp = Preprocessor({
            "cleaning": {"remove_html": True, "normalize_whitespace": True, "min_content_length": 10},
            "deduplication": {"enabled": True, "method": "exact"},
            "pii_masking": {"enabled": True, "replacement": "REDACTED"},
        })

    def test_clean_html(self):
        doc = Document(id="1", content="<p>Hello <b>world</b></p>")
        result = self.pp.process(doc)
        assert result is not None
        assert "<" not in result.content

    def test_normalize_whitespace(self):
        doc = Document(id="2", content="Hello    world\n\n\ntest   content here")
        result = self.pp.process(doc)
        assert "    " not in result.content

    def test_dedup(self):
        doc1 = Document(id="3", content="This is duplicate content for testing")
        doc2 = Document(id="4", content="This is duplicate content for testing")
        r1 = self.pp.process(doc1)
        r2 = self.pp.process(doc2)
        assert r1 is not None
        assert r2 is None  # Duplicate

    def test_pii_masking_ssn(self):
        doc = Document(id="5", content="My SSN is 123-45-6789 and I live here")
        result = self.pp.process(doc)
        assert "123-45-6789" not in result.content
        assert "[REDACTED]" in result.content

    def test_pii_masking_email(self):
        doc = Document(id="6", content="Contact me at user@example.com please today")
        result = self.pp.process(doc)
        assert "user@example.com" not in result.content

    def test_min_length_filter(self):
        doc = Document(id="7", content="Short")
        result = self.pp.process(doc)
        assert result is None


class TestChunker:
    def setup_method(self):
        self.chunker = Chunker({
            "strategy": "fixed",
            "chunk_size": 100,
            "chunk_overlap": 20,
        })

    def test_fixed_chunks(self):
        doc = Document(id="1", content="A" * 250)
        chunks = self.chunker.chunk(doc)
        assert len(chunks) >= 2
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_sentence_chunks(self):
        chunker = Chunker({"strategy": "sentence", "chunk_size": 50})
        doc = Document(id="2", content="First sentence. Second sentence. Third sentence. Fourth one.")
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_semantic_chunks(self):
        chunker = Chunker({"strategy": "semantic", "chunk_size": 50})
        doc = Document(id="3", content="Paragraph one content.\n\nParagraph two content.\n\nParagraph three.")
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_empty_content(self):
        doc = Document(id="4", content="")
        chunks = self.chunker.chunk(doc)
        assert len(chunks) == 0

    def test_chunk_metadata(self):
        doc = Document(id="5", content="A" * 250, metadata={"source": "test"})
        chunks = self.chunker.chunk(doc)
        assert all("source" in c.metadata for c in chunks)
        assert all("chunk_index" in c.metadata for c in chunks)


class TestEnricher:
    def test_keywords(self):
        enricher = Enricher({"generate_keywords": True, "add_timestamps": True})
        chunk = Chunk(id="1", content="Machine learning uses neural networks for pattern recognition in data science", document_id="doc1")
        result = enricher.enrich(chunk)
        assert "keywords" in result.metadata
        assert len(result.metadata["keywords"]) > 0

    def test_entities(self):
        enricher = Enricher({"extract_entities": True, "add_timestamps": True})
        chunk = Chunk(id="2", content="John Smith works at Google Research in Mountain View", document_id="doc1")
        result = enricher.enrich(chunk)
        assert "entities" in result.metadata

    def test_word_count(self):
        enricher = Enricher({"add_timestamps": True})
        chunk = Chunk(id="3", content="one two three four five", document_id="doc1")
        result = enricher.enrich(chunk)
        assert result.metadata["word_count"] == 5


class TestFileSystemLoader:
    def test_load_text_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "test1.txt").write_text("Hello world content here")
            (Path(tmpdir) / "test2.md").write_text("# Markdown\nSome content")
            (Path(tmpdir) / "test3.py").write_text("print('hello')")  # Not in defaults
            
            loader = FileSystemLoader(tmpdir, formats=["txt", "md"])
            docs = list(loader.load())
            assert len(docs) == 2
            assert all(isinstance(d, Document) for d in docs)


# ── Security Tests ──────────────────────────────────────────────────────────

class TestSecurity:
    def test_api_key_generation(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            mgr = APIKeyManager(f.name)
            key = mgr.generate_key("test", "user")
            assert key.startswith("llmragv3_")
            assert mgr.validate_key(key) is not None
            os.unlink(f.name)

    def test_api_key_revocation(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            mgr = APIKeyManager(f.name)
            key = mgr.generate_key("test", "user")
            mgr.revoke_key(key)
            assert mgr.validate_key(key) is None
            os.unlink(f.name)

    def test_rbac_permissions(self):
        from src.security.engine import RBACManager
        rbac = RBACManager({"enabled": True})
        assert rbac.check_permission("admin", "write") is True
        assert rbac.check_permission("reader", "write") is False
        assert rbac.check_permission("user", "query") is True

    def test_compliance_check(self):
        from src.security.engine import ComplianceManager
        cm = ComplianceManager({"frameworks": ["HIPAA"], "data_retention_days": 90, "right_to_delete": True})
        result = cm.run_compliance_check({
            "security": {
                "encryption": {"at_rest": True, "in_transit": True},
                "audit": {"enabled": True},
                "authentication": {"enabled": True},
                "compliance": {"data_retention_days": 90, "right_to_delete": True},
            },
            "ingestion": {"preprocessing": {"pii_masking": {"enabled": True}}},
        })
        assert "HIPAA" in result
        assert result["HIPAA"]["score"] > 0


# ── Integration Tests ───────────────────────────────────────────────────────

class TestIngestionPipeline:
    def test_full_pipeline_with_temp_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test documents
            docs_dir = Path(tmpdir) / "documents"
            docs_dir.mkdir()
            (docs_dir / "doc1.txt").write_text(
                "Machine learning is a subset of artificial intelligence. "
                "It uses algorithms to learn from data and make predictions. "
                "Deep learning is a type of machine learning using neural networks."
            )
            (docs_dir / "doc2.txt").write_text(
                "Natural language processing enables computers to understand human language. "
                "NLP techniques include tokenization, parsing, and sentiment analysis. "
                "Modern NLP relies heavily on transformer architectures."
            )

            config = {
                "sources": [{"type": "filesystem", "path": str(docs_dir), "formats": ["txt"], "recursive": True}],
                "preprocessing": {
                    "cleaning": {"remove_html": True, "normalize_whitespace": True, "min_content_length": 20},
                    "deduplication": {"enabled": True},
                    "pii_masking": {"enabled": False},
                },
                "chunking": {"strategy": "semantic", "chunk_size": 200, "chunk_overlap": 20},
                "enrichment": {"generate_keywords": True, "add_timestamps": True},
            }

            engine = IngestionEngine(config)
            chunks = engine.run()
            
            assert len(chunks) > 0
            assert all(isinstance(c, Chunk) for c in chunks)
            assert all("keywords" in c.metadata for c in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
