# 🧬 LLMRAG — Automated LLM & RAG Deployment Framework

<p align="center">
  <strong>AI/AGI/GI Genetic Frameworks Pipeline Automation</strong><br>
  <em>One-command deployment of production-grade RAG systems</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Docker-Ready-blue?logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=github" alt="CI/CD">
  <img src="https://img.shields.io/badge/Tests-22%20Passing-brightgreen" alt="Tests">
</p>

---

## 📖 What is LLMRAG?

**LLMRAG** is a complete, automated framework for deploying production-grade **Large Language Model (LLM)** and **Retrieval-Augmented Generation (RAG)** pipelines. It is designed for AI/AGI/GI genetic framework workflows — pulling data from multiple sources, embedding and indexing it, then exposing an intelligent query interface that retrieves relevant context and generates grounded answers with citations.

The framework handles the entire lifecycle:

```
Data Sources → Ingestion → Preprocessing → Chunking → Enrichment
    → Embedding → Vector Storage → Query Understanding → Retrieval
    → Post-Retrieval Reranking → Context Assembly → LLM Generation
    → Cited Output
```

All secured with authentication, RBAC, encryption, audit logging, and compliance checks.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LLMRAG PIPELINE ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────── DATA + INGESTION ──────────────────┐             │
│  │                                                         │             │
│  │  Filesystem ──┐                                         │             │
│  │  REST APIs ───┤   Preprocessing    Chunking   Enrichment│             │
│  │  Databases ───┼──▶ • Cleaning    ──▶ • Semantic ──▶ • Metadata    │  │
│  │  S3 Storage ──┤   • Dedup          • Fixed       • Keywords    │  │
│  │  Web Crawler ─┘   • PII Masking    • Recursive   • Entities    │  │
│  │                   • HTML Strip      • Sentence    • Summaries   │  │
│  └─────────────────────────────────────────────────────────┘             │
│                              │                                           │
│  ┌─────────────── EMBEDDING + STORAGE ───────────────────┐             │
│  │                                                         │             │
│  │  Embedding Models         Vector Databases              │             │
│  │  • OpenAI                 • ChromaDB (default)          │             │
│  │  • HuggingFace            • Pinecone                    │             │
│  │  • Cohere                 • pgvector                    │             │
│  │  • Local Models           • Qdrant                      │             │
│  │                           + Redis Cache                  │             │
│  └─────────────────────────────────────────────────────────┘             │
│                              │                                           │
│  ┌─────────────── RETRIEVAL PIPELINE ────────────────────┐             │
│  │                                                         │             │
│  │  User Query ──▶ Query Understanding ──▶ Multi-Search    │             │
│  │  • Question      • Rewrite               • Vector       │             │
│  │  • History        • Entity Detection      • Keyword      │             │
│  │  • Filters        • Intent Classification • Hybrid       │             │
│  │                   • Query Expansion                      │             │
│  │                                                         │             │
│  │  Post-Retrieval: Reranking → Dedup → Compression        │             │
│  │                  → Chunk Stitching → Permission Filter   │             │
│  └─────────────────────────────────────────────────────────┘             │
│                              │                                           │
│  ┌─────────────── GENERATION + OUTPUT ───────────────────┐             │
│  │                                                         │             │
│  │  Context Assembly ──▶ Prompt Construction ──▶ LLM       │             │
│  │  • Top chunks          • System prompt        • OpenAI   │             │
│  │  • Token budget         • User query           • Claude  │             │
│  │  • Citation mapping     • Context              • Ollama  │             │
│  │                        • Guardrails            • Local   │             │
│  │                                                         │             │
│  │  Output: Answer + Citations + Sources + Confidence       │             │
│  └─────────────────────────────────────────────────────────┘             │
│                                                                         │
│  ┌──── SECURITY ────┐  ┌──── OBSERVABILITY ────┐                       │
│  │ • API Key Auth    │  │ • JSON Logging         │                       │
│  │ • RBAC            │  │ • Audit Trail          │                       │
│  │ • PII Masking     │  │ • Metrics              │                       │
│  │ • Encryption      │  │ • RAG Evaluation       │                       │
│  │ • Compliance      │  │ • Feedback Loop        │                       │
│  │   (HIPAA/SOC2/    │  └──────────────────────────┘                       │
│  │    GDPR/FedRAMP)  │                                                   │
│  └───────────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start (5 Minutes)

### Option A: Clone & Run

```bash
# 1. Clone the repo
git clone https://github.com/kennyb7322/LLMRAG-V3.git
cd LLMRAG

# 2. Run setup
bash scripts/setup.sh

# 3. Set your API key
export OPENAI_API_KEY=sk-your-key-here

# 4. Add documents to data/documents/

# 5. Ingest
python main.py ingest

# 6. Query
python main.py query "What is machine learning?"

# 7. Start API server
python main.py serve
```

### Option B: Docker

```bash
# Clone
git clone https://github.com/kennyb7322/LLMRAG-V3.git
cd LLMRAG

# Set environment
cp .env.example .env
# Edit .env with your API keys

# Build and run
docker-compose up -d

# API available at http://localhost:8000
curl http://localhost:8000/health
```

### Option C: Use Ollama (100% Local — No API Keys)

```bash
git clone https://github.com/kennyb7322/LLMRAG-V3.git
cd LLMRAG

# Start with local LLM
docker-compose --profile local-llm up -d

# Update config to use Ollama
# In configs/pipeline_config.yaml:
#   generation.llm.provider: "ollama"
#   embedding.model.provider: "huggingface"

python main.py ingest
python main.py query "Your question"
```

---

## 📁 Repository Structure

```
LLMRAG/
├── main.py                      # Entry point — run everything from here
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container build
├── docker-compose.yml           # Full stack (API + Redis + Ollama)
├── Makefile                     # Common commands
├── .env.example                 # Environment template
│
├── configs/
│   └── pipeline_config.yaml     # Master configuration (all parameters)
│
├── src/
│   ├── ingestion/
│   │   └── engine.py            # Data loaders, preprocessor, chunker, enricher
│   ├── embedding/
│   │   └── engine.py            # Embedding models + vector store abstraction
│   ├── retrieval/
│   │   └── engine.py            # Query understanding, search, post-retrieval
│   ├── generation/
│   │   └── engine.py            # Context assembly, LLM calls, output
│   ├── security/
│   │   └── engine.py            # Auth, RBAC, encryption, audit, compliance
│   ├── pipeline/
│   │   ├── orchestrator.py      # Main pipeline — ties everything together
│   │   └── api_server.py        # FastAPI REST API
│   ├── tools/
│   │   ├── cli.py               # Command-line interface
│   │   ├── web_crawler.py       # Web content crawler
│   │   └── evaluator.py         # RAG quality evaluation
│   └── utils/
│       ├── config.py            # YAML config loader
│       └── logger.py            # Structured logging
│
├── scripts/
│   └── setup.sh                 # Bootstrap setup script
│
├── tests/
│   └── test_pipeline.py         # 22 unit + integration tests
│
├── data/
│   └── documents/               # Drop your files here
│
├── docs/
│   └── images/                  # Architecture diagrams
│
└── .github/
    └── workflows/
        └── ci.yml               # CI/CD pipeline (lint → test → build → deploy)
```

---

## 🔧 Configuration Reference

All pipeline behavior is controlled by a single YAML file: `configs/pipeline_config.yaml`.

### Data Sources

| Source | Config Key | Supported Formats |
|--------|-----------|-------------------|
| Filesystem | `ingestion.sources[].type: filesystem` | PDF, DOCX, TXT, MD, HTML, CSV, JSON |
| REST API | `ingestion.sources[].type: api` | JSON responses |
| Database | `ingestion.sources[].type: database` | PostgreSQL, MySQL, MSSQL, Oracle |
| AWS S3 | `ingestion.sources[].type: s3` | Same as filesystem |
| Web Crawler | `ingestion.sources[].type: web_crawler` | HTML pages |

### Embedding Providers

| Provider | Config Value | Models |
|----------|-------------|--------|
| OpenAI | `openai` | text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002 |
| HuggingFace | `huggingface` | all-MiniLM-L6-v2, all-mpnet-base-v2, any sentence-transformer |
| Cohere | `cohere` | embed-english-v3.0, embed-multilingual-v3.0 |
| Local | `local` | Any ONNX/PyTorch model on disk |

### Vector Databases

| Database | Config Value | Best For |
|----------|-------------|----------|
| ChromaDB | `chromadb` | Local dev, prototyping, small-medium datasets |
| Pinecone | `pinecone` | Production, managed service, large-scale |
| pgvector | `pgvector` | PostgreSQL users, existing DB infrastructure |
| Qdrant | `qdrant` | High-performance, filtering, hybrid search |

### LLM Providers

| Provider | Config Value | Models |
|----------|-------------|--------|
| OpenAI | `openai` | gpt-4o, gpt-4o-mini, gpt-4-turbo |
| Anthropic | `anthropic` | claude-sonnet-4-20250514, claude-opus-4-20250514 |
| Ollama | `ollama` | llama3, mistral, mixtral, phi3, gemma (local) |
| Local | `local` | Any GGUF model via llama-cpp-python |

### Chunking Strategies

| Strategy | Config Value | Description |
|----------|-------------|-------------|
| Semantic | `semantic` | Split on paragraph boundaries, merge small paragraphs |
| Recursive | `recursive` | Hierarchical splitting: paragraphs → sentences → words |
| Sentence | `sentence` | Split on sentence boundaries |
| Fixed | `fixed` | Fixed character windows with overlap |

---

## 🔌 CLI Reference

```bash
# Setup (first time)
python main.py setup

# Ingest data
python main.py ingest

# Query
python main.py query "How does transformer attention work?"

# Start API server
python main.py serve --port 8000 --workers 4

# Health check
python main.py health

# Compliance report
python main.py compliance

# Generate API key
python main.py generate-key --name "myapp" --role admin
```

---

## 🌐 API Reference

### Start the Server

```bash
python main.py serve
# or
make serve
```

### Endpoints

#### `POST /query`
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"question": "What is RAG?", "chat_history": []}'
```

Response:
```json
{
  "answer": "RAG (Retrieval-Augmented Generation) is a technique that...",
  "citations": [
    {"ref": "[1]", "source": "rag_overview.pdf", "score": 0.92}
  ],
  "sources": ["rag_overview.pdf"],
  "confidence": 0.87,
  "tokens_used": 1240
}
```

#### `POST /ingest`
```bash
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: your-admin-key"
```

#### `GET /health`
```bash
curl http://localhost:8000/health
```

#### `GET /compliance`
```bash
curl http://localhost:8000/compliance \
  -H "X-API-Key: your-key"
```

#### `GET /stats`
```bash
curl http://localhost:8000/stats
```

---

## 🔐 Security Features

### Authentication
- API key-based authentication with role assignment
- Keys stored securely with generation/revocation support
- Environment variable support for CI/CD

### RBAC (Role-Based Access Control)
- **admin**: Full access (read, write, delete, configure, query, manage keys)
- **user**: Read and query access
- **reader**: Read-only access

### Data Protection
- **PII Masking**: Auto-detect and mask SSNs, credit cards, emails, phone numbers
- **Encryption at Rest**: HMAC-based field-level encryption
- **Encryption in Transit**: TLS/HTTPS (configure via reverse proxy)
- **Audit Logging**: Every query and response logged with timestamps

### Compliance Frameworks
Configure in `pipeline_config.yaml` → `security.compliance.frameworks`:
- **HIPAA** — Healthcare data protection
- **SOC2** — Service organization controls
- **GDPR** — EU data privacy
- **FedRAMP** — Federal cloud security
- **PCI-DSS** — Payment card data

Run a compliance check:
```bash
python main.py compliance
```

---

## 📊 Evaluation & Observability

### Built-in RAG Evaluation

```python
from src.tools.evaluator import RAGEvaluator

evaluator = RAGEvaluator()
scores = evaluator.evaluate(
    question="What is deep learning?",
    answer="Deep learning is a subset of ML using neural networks...",
    contexts=["Deep learning uses multi-layered neural networks..."],
    ground_truth="Deep learning is machine learning with neural networks."
)
print(f"Faithfulness: {scores.faithfulness:.2%}")
print(f"Relevancy: {scores.relevancy:.2%}")
print(f"Overall: {scores.overall:.2%}")
```

### Metrics
- **Faithfulness**: How well the answer is grounded in retrieved context
- **Relevancy**: How relevant the answer is to the question
- **Context Precision**: How precise the retrieved chunks are
- **Answer Correctness**: F1 score against ground truth (when available)

### Logging
- Structured JSON logs to `./logs/pipeline.log`
- Audit trail to `./logs/audit.log`
- Console output with timestamps and modules

---

## 🐳 Deployment Options

### Local Development
```bash
python main.py setup
python main.py ingest
python main.py serve
```

### Docker
```bash
docker-compose up -d
```

### Docker with Local LLM (Ollama)
```bash
docker-compose --profile local-llm up -d
# Then pull a model
docker exec llmrag-ollama ollama pull llama3
```

### Docker with Monitoring
```bash
docker-compose --profile monitoring up -d
# Prometheus at http://localhost:9090
```

### GitHub Actions CI/CD
The included `.github/workflows/ci.yml` pipeline:
1. **Lint** — flake8, black, isort
2. **Test** — pytest with coverage
3. **Build** — Docker image pushed to GHCR
4. **Deploy** — Configure your deployment target

---

## 🧩 Extending the Pipeline

### Add a New Data Source
1. Create a loader class in `src/ingestion/engine.py`
2. Implement the `load()` generator method yielding `Document` objects
3. Register it in `IngestionEngine._load_all_sources()`

### Add a New Vector Database
1. Add init, store, and search methods in `src/embedding/engine.py`
2. Register the provider name in `VectorStore.initialize()`

### Add a New LLM Provider
1. Add a `_call_<provider>()` method in `src/generation/engine.py`
2. Register in `GenerationEngine._call_llm()`

### Custom Prompt Templates
Edit `configs/pipeline_config.yaml` → `generation.prompt_template`.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test class
pytest tests/test_pipeline.py::TestChunker -v
```

Current: **22 tests passing** covering config, preprocessing, chunking, enrichment, file loading, security, RBAC, compliance, and integration.

---

## 📋 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | If using OpenAI | OpenAI API key |
| `ANTHROPIC_API_KEY` | If using Anthropic | Anthropic API key |
| `COHERE_API_KEY` | If using Cohere | Cohere API key |
| `PINECONE_API_KEY` | If using Pinecone | Pinecone API key |
| `LLMRAG_API_KEY` | Optional | Pipeline API key for auth |
| `LLMRAG_CONFIG` | Optional | Path to config file |
| `LLMRAG_ENCRYPTION_KEY` | Optional | Encryption key for data at rest |
| `PORT` | Optional | API server port (default: 8000) |

---

## 🗺️ Roadmap

- [x] Multi-source data ingestion (filesystem, API, DB, S3, web)
- [x] Configurable chunking strategies (semantic, recursive, sentence, fixed)
- [x] Multi-provider embedding (OpenAI, HuggingFace, Cohere, local)
- [x] Multi-provider LLM (OpenAI, Anthropic, Ollama, local)
- [x] Vector store abstraction (ChromaDB, Pinecone, pgvector, Qdrant)
- [x] Query understanding (rewrite, entity detection, intent, expansion)
- [x] Post-retrieval processing (reranking, dedup, compression, stitching)
- [x] Security (auth, RBAC, PII masking, encryption, audit)
- [x] Compliance checks (HIPAA, SOC2, GDPR, FedRAMP)
- [x] FastAPI REST API with OpenAPI docs
- [x] Docker + Docker Compose deployment
- [x] GitHub Actions CI/CD
- [x] RAG evaluation metrics
- [ ] Streaming responses (SSE)
- [ ] Multi-modal RAG (images, tables)
- [ ] Graph RAG (knowledge graphs)
- [ ] Fine-tuning pipeline integration
- [ ] Kubernetes Helm chart
- [ ] Admin dashboard UI

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Ken Barnes, Ph.D.** — Founder & CTO/CISO, UCS Solutions

- 30+ years enterprise architecture and security consulting
- Patent-pending AGI Olfactory Interface Technology
- CISSP, CISM, CCSP, CRISC, CEH, TOGAF, SABSA, PMP, Six Sigma Black Belt

---

<p align="center">
  <strong>Built with 🧬 by UCS Solutions</strong><br>
  <em>github.com/kennyb7322/LLMRAG-V3</em>
</p>
