"""
LLMRAG Command-Line Interface
================================
Run pipeline operations directly from the terminal.

Usage:
    python -m src.tools.cli ingest
    python -m src.tools.cli query "What is machine learning?"
    python -m src.tools.cli serve
    python -m src.tools.cli health
    python -m src.tools.cli compliance
    python -m src.tools.cli setup
    python -m src.tools.cli generate-key --name "myapp" --role user
"""

import sys
import os
import json
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def cmd_ingest(args):
    """Run the ingestion pipeline."""
    from src.pipeline.orchestrator import LLMRAGPipeline
    pipeline = LLMRAGPipeline(args.config)
    stats = pipeline.ingest()
    print(f"\n✅ Ingestion complete:")
    print(f"   Chunks created: {stats.chunks_created}")
    print(f"   Vectors stored: {stats.vectors_stored}")
    print(f"   Time: {stats.ingestion_time_sec + stats.embedding_time_sec:.1f}s")


def cmd_query(args):
    """Run a RAG query."""
    from src.pipeline.orchestrator import LLMRAGPipeline
    pipeline = LLMRAGPipeline(args.config)
    result = pipeline.query(args.question)
    
    print(f"\n{'═' * 60}")
    print(f"Question: {args.question}")
    print(f"{'─' * 60}")
    print(f"\n{result.answer}")
    print(f"\n{'─' * 60}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Sources: {', '.join(result.sources)}")
    print(f"Tokens: {result.tokens_used}")
    print(f"{'═' * 60}")


def cmd_serve(args):
    """Start the API server."""
    import uvicorn
    from src.pipeline.api_server import create_app
    
    os.environ["LLMRAG_CONFIG"] = args.config
    app = create_app()
    
    print(f"\n🚀 Starting LLMRAG API Server on port {args.port}")
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


def cmd_health(args):
    """Check pipeline health."""
    from src.pipeline.orchestrator import LLMRAGPipeline
    pipeline = LLMRAGPipeline(args.config)
    health = pipeline.health_check()
    print(json.dumps(health, indent=2))


def cmd_compliance(args):
    """Run compliance checks."""
    from src.pipeline.orchestrator import LLMRAGPipeline
    pipeline = LLMRAGPipeline(args.config)
    report = pipeline.compliance_report()
    
    if not report:
        print("No compliance frameworks configured.")
        return
    
    for framework, data in report.items():
        score = data.get("score", 0)
        print(f"\n{framework}: {score:.0f}%")
        for p in data.get("passed", []):
            print(f"  ✅ {p}")
        for f in data.get("failed", []):
            print(f"  ❌ {f}")


def cmd_generate_key(args):
    """Generate an API key."""
    from src.security.engine import APIKeyManager
    mgr = APIKeyManager()
    key = mgr.generate_key(name=args.name, role=args.role)
    print(f"\n🔑 API Key generated:")
    print(f"   Name: {args.name}")
    print(f"   Role: {args.role}")
    print(f"   Key:  {key}")
    print(f"\nSet as environment variable:")
    print(f"   export LLMRAG_API_KEY={key}")


def cmd_setup(args):
    """Interactive setup wizard."""
    print("\n╔══════════════════════════════════════════════╗")
    print("║     LLMRAG Setup Wizard                    ║")
    print("╚══════════════════════════════════════════════╝\n")

    # Create directories
    dirs = ["data/documents", "data/vectordb", "logs", "models/embeddings", "models/llm"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  📁 Created: {d}/")

    # Check dependencies
    print("\n📦 Checking dependencies...")
    deps = {
        "yaml": "pyyaml",
        "chromadb": "chromadb",
        "openai": "openai",
        "fastapi": "fastapi[standard]",
        "sentence_transformers": "sentence-transformers",
    }
    missing = []
    for mod, pkg in deps.items():
        try:
            __import__(mod)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} — pip install {pkg}")
            missing.append(pkg)

    if missing:
        print(f"\nInstall missing: pip install {' '.join(missing)}")

    print("\n✅ Setup complete! Next steps:")
    print("  1. Add documents to data/documents/")
    print("  2. Set API keys: export OPENAI_API_KEY=sk-...")
    print("  3. Run: python -m src.tools.cli ingest")
    print("  4. Query: python -m src.tools.cli query 'Your question here'")
    print("  5. Serve: python -m src.tools.cli serve")


def main():
    parser = argparse.ArgumentParser(
        prog="llmrag",
        description="LLMRAG — Automated LLM & RAG Pipeline CLI",
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/pipeline_config.yaml",
        help="Path to config file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # ingest
    subparsers.add_parser("ingest", help="Run ingestion pipeline")

    # query
    p_query = subparsers.add_parser("query", help="Run a RAG query")
    p_query.add_argument("question", help="The question to ask")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--workers", type=int, default=4)

    # health
    subparsers.add_parser("health", help="Check pipeline health")

    # compliance
    subparsers.add_parser("compliance", help="Run compliance checks")

    # generate-key
    p_key = subparsers.add_parser("generate-key", help="Generate API key")
    p_key.add_argument("--name", required=True)
    p_key.add_argument("--role", default="user", choices=["admin", "user", "reader"])

    # setup
    subparsers.add_parser("setup", help="Interactive setup wizard")

    args = parser.parse_args()

    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "serve": cmd_serve,
        "health": cmd_health,
        "compliance": cmd_compliance,
        "generate-key": cmd_generate_key,
        "setup": cmd_setup,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
